import jax
import jax.numpy as jnp
from flax import struct
from jaxav.utils.math import quat_to_matrix, quat_to_euler, normalize

@struct.dataclass
class RateControllerParam:
    max_rotation_velocities: jnp.ndarray
    max_thrust: jnp.ndarray
    ang_gain: jnp.ndarray
    ang_acc_to_rotor_vel: jnp.ndarray


def _compute_allocation_matrix(params):
    return

class RateController:

    def init(self, params):
        inertia = jnp.array([
            params["inertia"]["xx"], params["inertia"]["yy"], params["inertia"]["zz"]]
        )
        arm_lengths = jnp.array(params["rotor_configuration"]["arm_lengths"])
        rotor_angles = jnp.array(params["rotor_configuration"]["rotor_angles"])
        rotor_directions = jnp.array(params["rotor_configuration"]["directions"])
        A = jnp.stack([
            jnp.sin(rotor_angles), 
            -jnp.cos(rotor_angles),
            -rotor_directions,
            jnp.ones_like(rotor_directions)
        ])
        arm_length = arm_lengths.mean()
        kf = jnp.array(params["rotor_configuration"]["force_constants"]).mean()
        km = jnp.array(params["rotor_configuration"]["moment_constants"]).mean()
        max_rotation_velocities = jnp.array(params["rotor_configuration"]["max_rotation_velocities"])
        max_thrust = max_rotation_velocities**2 * kf
        K = jnp.diag(1 /jnp.array([arm_length * kf, arm_length * kf, km, kf]))
        I = jnp.diag(jnp.ones(4).at[:3].set(inertia))

        ang_gain = jnp.array([0.52, 0.52, 0.025]) / inertia
        ang_acc_to_rotor_vel = A.T @ jnp.linalg.inv(A @ A.T) @ K @ I
        return RateControllerParam(
            max_rotation_velocities=max_rotation_velocities,
            max_thrust=max_thrust,
            ang_gain=ang_gain, 
            ang_acc_to_rotor_vel=ang_acc_to_rotor_vel
        )
    
    def __call__(self, drone_state, control_ref, params: RateControllerParam):
        angvel = drone_state.angvel
        error = angvel - control_ref[:3]
        acc_des = (
            - error * params.ang_gain
            + jnp.cross(angvel, angvel) # gyroscopic term?
        )
        angacc_thrust = control_ref.at[:3].set(acc_des)
        rotor_vel = params.ang_acc_to_rotor_vel @ angacc_thrust
        rotor_vel = jnp.clip(rotor_vel, 0)
        # rotor_vel = jnp.sqrt(rotor_vel)
        return (rotor_vel / params.max_rotation_velocities) * 2 - 1


@struct.dataclass
class PositionControllerParams:
    max_rotation_velocities: jnp.ndarray
    max_thrust: jnp.ndarray
    pos_gain: jnp.ndarray
    vel_gain: jnp.ndarray
    att_gain: jnp.ndarray
    ang_gain: jnp.ndarray
    alloc_matrix: jnp.ndarray
    last_rpy: jnp.ndarray

class PositionController:

    def init(self, params):
        inertia = jnp.array([
            params["inertia"]["xx"], params["inertia"]["yy"], params["inertia"]["zz"]]
        )
        arm_lengths = jnp.array(params["rotor_configuration"]["arm_lengths"])
        rotor_angles = jnp.array(params["rotor_configuration"]["rotor_angles"])
        rotor_directions = jnp.array(params["rotor_configuration"]["directions"])
        max_rotation_velocities = jnp.array(params["rotor_configuration"]["max_rotation_velocities"])
        kf = jnp.array(params["rotor_configuration"]["force_constants"])
        km = jnp.array(params["rotor_configuration"]["moment_constants"])
        max_thrust = max_rotation_velocities**2 * kf
        A = jnp.stack([
            jnp.sin(rotor_angles) * arm_lengths,
            -jnp.cos(rotor_angles) * arm_lengths,
            -rotor_directions * km / kf,
            jnp.ones_like(rotor_angles),
        ])

        I = jnp.diag(jnp.ones(4).at[:3].set(inertia))
        alloc_matrix = A.T @ jnp.linalg.inv(A @ A.T) @ I
        # pos_gain = jnp.array([6., 6., 6.])
        # vel_gain = jnp.array([4.7, 4.7, 4.7])
        # att_gain = jnp.array([3, 3, .15]) / inertia
        # ang_gain = jnp.array([.52, .52, .18]) / inertia
        pos_gain = jnp.array([4., 4., 4.])
        vel_gain = jnp.array([2., 2., 2.])
        att_gain = jnp.array([0.7, 0.7, .035]) / inertia
        ang_gain = jnp.array([.1, .1, .025]) / inertia
        return PositionControllerParams(
            max_rotation_velocities=max_rotation_velocities,
            max_thrust=max_thrust,
            pos_gain=pos_gain,
            vel_gain=vel_gain,
            att_gain=att_gain,
            ang_gain=ang_gain,
            alloc_matrix=alloc_matrix,
            last_rpy=jnp.zeros(3)
        )

    def __call__(
        self, 
        drone_state, 
        control_ref, 
        params: PositionControllerParams
    ):
        pos = drone_state.pos
        rot = drone_state.rot
        vel = drone_state.vel
        angvel = drone_state.angvel
        pos_target = control_ref[:3] 
        vel_target = control_ref[3:6] 
        yaw_target = control_ref[6]

        # compute desired acceleration
        pos_error = pos_target - pos
        vel_error = vel_target - vel
        acc_des = (
            pos_error * params.pos_gain
            + vel_error * params.vel_gain
            + jnp.array([0., 0., 9.81])
        )

        # compute desired angular acceleration
        R = quat_to_matrix(rot)
        b1_des = jnp.array([jnp.cos(yaw_target), jnp.sin(yaw_target), 0])
        b3_des = normalize(acc_des)[0]
        b2_des = normalize(jnp.cross(b3_des, b1_des))[0]
        R_des = jnp.stack([
            jnp.cross(b2_des, b3_des),
            b2_des,
            b3_des
        ], 1)
        ang_error_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
        ang_error = jnp.array(
            [ang_error_matrix[2, 1], ang_error_matrix[0, 2], ang_error_matrix[1, 0]]
        )
        angrate_error = angvel
        angacc_des = (
            - ang_error * params.att_gain
            - angrate_error * params.ang_gain
            + jnp.cross(angvel, angvel)
        )

        thrust = drone_state.mass * (acc_des @ R[:, 2])
        angacc_thrust = (
            jnp.zeros(4)
            .at[:3].set(angacc_des)
            .at[3].set(thrust)
        )
        # print(angacc_thrust)
        rotor_vel = params.alloc_matrix @ angacc_thrust
        rotor_vel = jnp.clip(rotor_vel, 0)
        # rotor_vel = jnp.sqrt(rotor_vel)
        return (rotor_vel / params.max_thrust) * 2 - 1
