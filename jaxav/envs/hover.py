from jaxav.base import EnvBase, EnvState as _EnvState
from jaxav.dynamics import DroneState, step
from jaxav.math import euler_to_quat, lerp

import jax
import jax.numpy as jnp
import distrax
from flax import struct
from jax import lax
from gymnax.environments.spaces import Box
from typing import Any

@struct.dataclass
class EnvState(_EnvState):
    drone: DroneState = None

class Hover(EnvBase):
    def __init__(
        self, 
        drone_model: str,
    ):
        self.template_state = DroneState.load(drone_model)
        self.target_pos = jnp.array([0., 0., 2.5])
        self.observation_space = Box(-jnp.inf, jnp.inf, (17,))
        self.action_space = Box(-1, 1, (4,))
    
    def init(self, key):
        return None

    def reset(self, params, key: jax.random.KeyArray):
        init_pos_dist = distrax.Uniform(
            jnp.array([-2., -2., 0.]),
            jnp.array([2., 2., 2.5])
        )
        init_rpy_dist = distrax.Uniform(
            low=jnp.array([-.1, -.1, 0.]),
            high=jnp.array([.1, .1, 2.])
        )
        drone_state = self.template_state.replace(
            pos=init_pos_dist.sample(seed=key),
            rot=euler_to_quat(init_rpy_dist.sample(seed=key))
        )
        env_state = EnvState(
            drone=drone_state, 
            max_episode_len=500,
            metrics={
                "pos_error": jnp.array([0.]), 
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
            env_state.drone.rot,
            env_state.drone.vel,
            env_state.drone.angvel,
            jnp.full(4, env_state.step / env_state.max_episode_len)
        ])
        return obs
    
    def _reward_and_done(self, env_state: EnvState):
        distance = jnp.linalg.norm(self.target_pos - env_state.drone.pos, keepdims=True)
        reward = jnp.exp(-distance)
        done = (
            (env_state.step > env_state.max_episode_len)
            | (distance > 4.)
        )
        metrics = {
            "pos_error": lerp(env_state.metrics["pos_error"], distance, 0.8),
            "episode_len": env_state.step
        }
        return reward, done, metrics

