import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.typing import ArrayLike
from .utils.math import quat_mul, quat_rotate, normalize


@struct.dataclass
class Transform:
    pos: ArrayLike
    rot: ArrayLike

    @classmethod
    def zero(cls, shape):
        if isinstance(shape, int):
            shape = (shape, )
        return cls(
            pos=jnp.zeros((*shape, 3)),
            rot=jnp.zeros((*shape, 4)).at[..., 0].set(1.)
        )

    def __call__(self, vec: ArrayLike):
        shape = jnp.broadcast_shapes(vec.shape[:-1], self.rot.shape[:-1])
        pos = jnp.broadcast_to(self.pos, (*shape, 3)).reshape(-1, 3)
        rot = jnp.broadcast_to(self.rot, (*shape, 4)).reshape(-1, 4)
        vec = pos + jax.vmap(quat_rotate)(vec.reshape(-1, 3), rot)
        return vec.reshape(*shape, 3)


@struct.dataclass
class DroneState:
    # kinematic state
    pos: ArrayLike
    rot: ArrayLike
    vel: ArrayLike
    angvel: ArrayLike

    # params
    mass: ArrayLike
    inertia: ArrayLike
    kf: ArrayLike
    km: ArrayLike
    rotor_max_rot: ArrayLike
    rotor_trans: ArrayLike
    rotor_rot: ArrayLike 
    rotor_dir: ArrayLike

    # throttle
    a: ArrayLike
    thro: ArrayLike


    def thrust2weight(self, g):
        
        return 

    def __getitem__(self, index):
        return jax.tree_map(lambda x: x[index], self)


    @classmethod
    def load(cls, path: str):
        import yaml
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        mass = jnp.array(params["mass"])
        inertia = jnp.array([
            params["inertia"]["xx"], params["inertia"]["yy"], params["inertia"]["zz"]]
        )

        arm_lengths = jnp.array(params["rotor_configuration"]["arm_lengths"])
        rotor_angles = jnp.array(params["rotor_configuration"]["rotor_angles"])
        max_rot = jnp.array(params["rotor_configuration"]["max_rotation_velocities"])
        kf = jnp.array(params["rotor_configuration"]["force_constants"])
        km = jnp.array(params["rotor_configuration"]["moment_constants"])
        
        rotor_trans = (
            jnp.zeros((len(arm_lengths), 3))
            .at[:, :2]
            .set(jnp.stack([jnp.cos(rotor_angles), jnp.sin(rotor_angles)], -1))
            * arm_lengths[:, None]
        )
        rotor_rot = jnp.tile(jnp.array([1., 0., 0., 0.]), (len(arm_lengths), 1))
        rotor_dir = jnp.array(params["rotor_configuration"]["directions"])

        return cls(
            pos=jnp.zeros(3), rot=jnp.array([1., 0., 0., 0.]),
            vel=jnp.zeros(3), angvel=jnp.zeros(3),
            mass=mass, inertia=inertia,
            kf=kf, km=km,
            rotor_max_rot=max_rot,
            rotor_trans=rotor_trans,
            rotor_rot=rotor_rot,
            rotor_dir=rotor_dir,
            a=jnp.array(0.5), thro=jnp.zeros(len(arm_lengths))
        )

    @property
    def up(self):
        rot = self.rot.reshape(-1, 4)
        up = jax.vmap(quat_rotate, (None, 0))(jnp.array([0., 0., 1.]), rot)
        return jnp.reshape(up, (*self.rot.shape[:-1], 3))
    
    @property
    def heading(self):
        rot = self.rot.reshape(-1, 4)
        heading = jax.vmap(quat_rotate, (None, 0))(jnp.array([1., 0., 0.]), rot)
        return jnp.reshape(heading, (*self.rot.shape[:-1], 3))


def step(state: DroneState, target_thro: ArrayLike, dt: float=0.02):
    if not target_thro.ndim == 1:
        raise ValueError

    target_thro = jnp.sqrt((jnp.clip(target_thro, -1, 1) + 1) / 2)
    thro = state.thro + state.a * (target_thro - state.thro)

    tmp = (thro * state.rotor_max_rot) ** 2
    thrust = tmp * state.kf
    force_body = jnp.zeros((thro.shape[0], 3)).at[:, 2].set(thrust)
    force_world = jax.vmap(quat_rotate)(force_body, state.rotor_rot)

    moment = tmp * state.km * -state.rotor_dir
    torque = jnp.zeros((thro.shape[0], 3)).at[:, 2].set(moment)
    torque = jax.vmap(quat_rotate)(torque, state.rotor_rot)
    torque += jnp.cross(force_body, -state.rotor_trans)

    g = jnp.array([0., 0., -9.8])
    acc = quat_rotate(jnp.sum(force_world, 0), state.rot) / state.mass + g
    angacc = jnp.sum(torque, 0) / state.inertia

    vel = state.vel + dt * acc
    angvel = state.angvel + dt * angacc

    pos = state.pos + dt * vel
    rot = normalize(state.rot + dt * 0.5 * quat_mul(state.rot, jnp.zeros(4).at[1:].set(angvel)))[0]
    # rot = normalize(state.rot + dt * 0.5 * quat_mul(state.rot, jnp.ones(4).at[1:].set(angvel)))[0]

    return state.replace(
        pos=pos, rot=rot,
        vel=vel, angvel=angvel,
        thro=thro
    )

