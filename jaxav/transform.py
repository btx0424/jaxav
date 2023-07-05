import jax
import jax.numpy as jnp

from typing import Sequence
from jaxav.base import EnvBase, EnvState

class EnvTransform:
    def __init__(self) -> None:
        self._parent = None
    
    def step(self, obs, action, reward, done, next_state: EnvState):
        return obs, reward, done, next_state
    
    def inv(self, state: EnvState, action):
        return state, action
    
    def reset(self, obs, state: EnvState):
        return obs, state

    @property
    def parent(self) -> EnvBase:
        return self._parent
    
    @parent.setter
    def parent(self, value: EnvBase):
        if not isinstance(value, EnvBase):
            raise TypeError
        self._parent = value


class Compose(EnvTransform):
    def __init__(self, transforms: Sequence[EnvTransform]) -> None:
        super().__init__()
        for transform in transforms:
            transform._parent = None
        self.transforms = transforms
    
    @property
    def parent(self) -> EnvBase:
        return self._parent
    
    @parent.setter
    def parent(self, value: EnvBase):
        for transform in self.transforms:
            transform.parent = value
        self._parent = value
    
    def step(self, obs, action, reward, done, next_state: EnvState):
        for transform in self.transforms:
            obs, reward, done, next_state = transform.step(
                obs, action, reward, done, next_state
            )
        return obs, reward, done, next_state
        
    def inv(self, state: EnvState, action):
        for transform in reversed(self.transforms):
            state, action = transform.inv(state, action)
        return state, action
    
    def reset(self, obs, state):
        return super().reset(obs, state)


class TransformedEnv(EnvBase):
    def __init__(self, env: EnvBase, transform: EnvTransform):
        self.base_env = env
        self.transform = transform
        self.transform.parent = self

    @property
    def action_space(self):
        return self.base_env.action_space
    
    def init(self, key: jax.random.KeyArray):
        return self.base_env.init(key)
    
    def step(self, state: EnvState, action):
        state, action = self.transform.inv(state, action)
        obs, reward, done, next_state = self.base_env.step(state, action)
        obs, reward, done, next_state = self.transform.step(
            obs, action, reward, done, next_state
        )
        return obs, reward, done, next_state
    
    def reset(self, env_params, key: jax.random.KeyArray):
        obs, state = self.base_env.reset(env_params, key)
        obs, state = self.transform.reset(obs, state)
        return obs, state


def _insert(a, item):
    return a.at[1:].set(a[:-1]).at[0].set(item)


class History(EnvTransform):

    def __init__(self, horizon: int=5):
        self.horizon = horizon

    def step(
        self, 
        obs, 
        action,
        reward, 
        done, 
        next_state: EnvState
    ):
        info = next_state.info
        obs_h = info["obs_h"] = _insert(info["obs_h"], obs)
        act_h = info["act_h"] = _insert(info["act_h"], action) 
        obs = jnp.concatenate([obs_h, act_h], axis=-1)
        return obs, reward, done, next_state
    
    def reset(self, obs, state):
        info = state.info
        obs_h = info["obs_h"] = jnp.zeros((self.horizon, obs.shape[-1])).at[0].set(obs)
        act_h = info["act_h"] = jnp.zeros((self.horizon, self.parent.action_space.shape[-1]))
        obs = jnp.concatenate([obs_h, act_h], axis=-1)
        return obs, state

