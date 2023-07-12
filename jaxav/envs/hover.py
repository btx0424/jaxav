from jaxav.base import EnvBase, EnvState as _EnvState
from jaxav.dynamics import DroneState, step
from jaxav.utils.math import euler_to_quat, lerp, uniform, heading

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax
from gymnax.environments.spaces import Box
from typing import Sequence
from jax.typing import ArrayLike

@struct.dataclass
class EnvState(_EnvState):
    drone: DroneState
    ref_heading: ArrayLike

class Hover(EnvBase):
    def __init__(
        self, 
        drone_model: str,
    ):
        self.template_state = DroneState.load(drone_model)
        self.target_pos = jnp.array([0., 0., 2.5])
        self.observation_space = Box(-jnp.inf, jnp.inf, (20,))
        self.action_space = Box(-1, 1, (4,))
    
    def init(self, key):

        return None

    def reset(self, params, key: jax.random.KeyArray):
        keys = jax.random.split(key, 3)
        init_pos = uniform(
            keys[0], jnp.array([-2., -2., 0.]), jnp.array([2., 2., 2.5])
        )
        init_rpy = uniform(
            keys[1], jnp.array([-.1, -.1, 0.]), jnp.array([.1, .1, 2.])
        ) * jnp.pi
        ref_rpy = uniform(
            keys[2], jnp.array([0., 0., 0.]), jnp.array([0., 0., 2.])
        ) * jnp.pi
        drone_state = self.template_state.replace(
            pos=init_pos,
            rot=euler_to_quat(init_rpy)
        )
        env_state = EnvState(
            drone=drone_state, 
            ref_heading=heading(euler_to_quat(ref_rpy)),
            max_episode_len=500,
            metrics={
                "pos_error": jnp.array(0.), 
                "heading": jnp.array(0.),
                "episode_len": 0
            }
        )
        obs = self._obs(env_state)
        return obs, env_state
    
    def step(self, env_state: EnvState, action):
        env_state = env_state.replace(
            drone=step(env_state.drone, action, 0.02),
            step=env_state.step + 1,
            is_init=False
        )
        obs = self._obs(env_state)
        reward, done, metrics = self._reward_and_done(env_state)
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
        obs = jnp.concatenate([
            self.target_pos - env_state.drone.pos,
            env_state.ref_heading-env_state.drone.heading,
            env_state.drone.rot,
            env_state.drone.vel,
            env_state.drone.angvel,
            jnp.full(4, env_state.step / env_state.max_episode_len)
        ])
        return obs
    
    def _reward_and_done(self, env_state: EnvState):
        distance = jnp.linalg.norm(self.target_pos - env_state.drone.pos)
        reward_pos = jnp.exp(-distance)
        reward_heading = jnp.dot(env_state.ref_heading, env_state.drone.heading)
        reward = (reward_pos + 0.5 * reward_heading)[None]
        done = (
            (env_state.step == env_state.max_episode_len)
            | (distance > 4.)
        )[None]
        metrics = {
            "pos_error": lerp(env_state.metrics["pos_error"], distance, 0.8),
            "heading": env_state.metrics["heading"] + reward_heading,
            "episode_len": env_state.step + 1
        }
        return reward, done, metrics

    def render_matplotlib(self, states: Sequence[EnvState]):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np
        from jaxav.utils.visualization import Drone, Traj3D

        states = jax.tree_map(np.array, states)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 4)
        ax.plot([0, 1], [0, 0], [0, 0])

        state = states[0]
        ref_pos = ax.scatter(*self.target_pos)
        ref_heading = np.stack([self.target_pos, self.target_pos+state.ref_heading], axis=-2)
        ref_heading_line = ax.plot(*ref_heading.T, "--")[0]
        drone = Drone(ax, state.drone)
        drone_traj = Traj3D(ax, state.drone.pos)

        def update(state: EnvState):
            drone.update(state.drone)
            drone_traj.update(state.drone.pos)
            
        anim = animation.FuncAnimation(
            fig, update, states[1:]
        )
        return anim
