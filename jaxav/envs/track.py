from jaxav.base import EnvBase, EnvState as _EnvState
from jaxav.dynamics import DroneState, step, Transform
from jaxav.utils.math import euler_to_quat, lerp, quat_rotate, uniform, sign, normalize

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as D

from flax import struct
from jax import lax
from gymnax.environments.spaces import Box
from jax.typing import ArrayLike
from typing import Sequence

@struct.dataclass
class EnvState(_EnvState):
    drone: DroneState 

    traj_w: ArrayLike
    traj_c: ArrayLike
    traj_q: ArrayLike # rotation
    traj_a: ArrayLike # scaling

    ref_pos: ArrayLike
    distance: ArrayLike
    heading: ArrayLike


def lemniscate(t: ArrayLike, c: ArrayLike):
    sin_t = jnp.sin(t)
    cos_t = jnp.cos(t)
    sin2p1 = jnp.square(sin_t) + 1
    x = jnp.stack([cos_t, sin_t * cos_t, c * sin_t], axis=-1) 
    x = x / sin2p1[..., None]
    return x


def scale_time(t: ArrayLike):
    return t / (1 + 1/jnp.abs(t))


class Track(EnvBase):
    def __init__(
        self, 
        drone_model: str,
    ):
        self.template_state = DroneState.load(drone_model)
        self.FUTURE_REF_STEPS = 5
        self.T0 = jnp.pi / 2.

        self.action_space = Box(-1, 1, (4,))
        self.observation_space = Box(
            -jnp.inf, jnp.inf, (16 + self.FUTURE_REF_STEPS,)
        )
        self.reset_thres = 1.
    
    def init(self, key):
        return None

    def reset(self, params, key: jax.random.KeyArray):
        keys = jax.random.split(key, 5)
        traj_w = uniform(keys[0], 0.8, 1.1) * sign(keys[0])
        traj_c = uniform(keys[1], -0.6, 0.6)
        traj_a = uniform(keys[2], jnp.array([1.8, 1.8, 1.]), jnp.array([3.2, 3.2, 1.5]))
        traj_q = euler_to_quat(
            uniform(keys[3], jnp.array([0., 0., 0.]), jnp.array([0., 0., 2*jnp.pi]))
        )
        pos = quat_rotate(lemniscate(self.T0, traj_c), traj_q) * traj_a
        rot = euler_to_quat(
            uniform(keys[4], jnp.array([0., 0., 0.]), jnp.array([0., 0., 2*jnp.pi]))
        )

        drone_state = self.template_state.replace(pos=pos, rot=rot)
        env_state = EnvState(
            drone=drone_state, 
            max_episode_len=800,
            traj_w=traj_w,
            traj_c=traj_c,
            traj_q=traj_q,
            traj_a=traj_a,
            metrics={
                "tracking_error": jnp.array(0.), 
                "heading": jnp.array(0.),
                "episode_len": 0,
            },
            ref_pos=pos,
            distance=0.,
            heading=drone_state.heading
        )
        obs, env_state = self._obs(env_state)

        return obs, env_state
    
    def step(self, env_state: EnvState, action):
        env_state = env_state.replace(
            drone=step(env_state.drone, action, env_state.dt),
            step=env_state.step + 1,
            is_init=False
        )
        
        obs, env_state = self._obs(env_state)
        distance = env_state.distance
        reward_tracking = jnp.exp(- 1.6 * distance)
        reward_heading = jnp.dot(
            env_state.heading[:2], 
            normalize(env_state.drone.heading[:2])[0]
        )
        reward = (reward_tracking + 0. * reward_heading)[None]
        done = (
            (env_state.step == env_state.max_episode_len)
            | (distance > self.reset_thres)
        )[None]
        metrics = {
            "tracking_error": lerp(env_state.metrics["tracking_error"], distance, 0.6),
            "episode_len": env_state.step + 1,
            "heading": env_state.metrics["heading"] + reward_heading
        }

        env_state = env_state.replace(
            Return=env_state.Return + reward,
            metrics=metrics
        )
        return (
            lax.stop_gradient(obs),
            reward, 
            done, 
            lax.stop_gradient(env_state)
        )

    def _obs(self, env_state: EnvState):
        t = (env_state.step + jnp.arange(5)) * env_state.dt * env_state.traj_w
        x = jax.vmap(lemniscate, in_axes=(0, None))(
            self.T0 + scale_time(t), env_state.traj_c
        )
        x = env_state.traj_a * jax.vmap(quat_rotate, in_axes=(0, None))(x, env_state.traj_q)
        traj_heading, _ = normalize(jnp.zeros(3).at[:2].set((x[1]-x[0])[:2]))
        obs = jnp.concatenate([
            jnp.reshape(x - env_state.drone.pos, -1),
            traj_heading-env_state.drone.heading,
            env_state.drone.rot,
            env_state.drone.vel,
            env_state.drone.angvel,
            jnp.full(4, env_state.step / env_state.max_episode_len)
        ])
        distance = jnp.linalg.norm(x[0] - env_state.drone.pos)

        return obs, env_state.replace(
            ref_pos=x[0],
            distance=distance, 
            heading=traj_heading
        )

    def render_matplotlib(self, states: Sequence[EnvState]):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np
        states = jax.tree_map(np.array, states)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 4)
        ax.plot([0, 1], [0, 0], [0, 0])

        state = states[0]

        transform = Transform(state.drone.pos, state.drone.rot)
        arms = np.stack([
            np.zeros_like(state.drone.rotor_trans), 
            state.drone.rotor_trans
        ], axis=-2)
        arms = transform(arms)

        arm_lines = [ax.plot(*arm.T)[0] for arm in arms]
        ref_pos = ax.scatter(*state.ref_pos)

        class Traj3D:
            def __init__(self, ax, x0):
                self.traj = [x0]
                self.line, = ax.plot(*x0.T)
            
            def update(self, x):
                self.traj.append(x)
                x, y, z = zip(*self.traj)
                self.line.set_data(x, y)
                self.line.set_3d_properties(z)
        
        drone_traj = Traj3D(ax, state.drone.pos)
        ref_traj = Traj3D(ax, state.ref_pos)

        def update(state: EnvState):
            transform = Transform(state.drone.pos, state.drone.rot)
            arms = np.stack([
                np.zeros_like(state.drone.rotor_trans), 
                state.drone.rotor_trans
            ], axis=-2)
            arms = transform(arms)
            for arm_line, arm in zip(arm_lines, arms):
                arm_line.set_data(*arm.T[:2])
                arm_line.set_3d_properties(arm.T[2])
            drone_traj.update(state.drone.pos)
            ref_traj.update(state.ref_pos)
            ref_pos.set_offsets(state.ref_pos[:2])
            ref_pos.set_3d_properties(state.ref_pos[2], zdir="z")
            
        anim = animation.FuncAnimation(
            fig, update, states[1:]
        )
        return anim




if __name__ == "__main__":
    import os
    drone = os.path.join(os.path.dirname(__file__), "../asset/hummingbird.yaml")
    env = Track(drone)

    key = jax.random.PRNGKey(0)
    params = env.init(key)
    obs, env_state = env.reset(params, key)
    print(env_state)