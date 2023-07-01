import jax
import jax.numpy as jnp

from flax import struct
from typing import Callable, Any


class EnvBase:
    def __init__(self) -> None:
        pass

    def step(self, state, action):
        pass

    def reset(self, env_params, key: jax.random.KeyArray):
        pass


@struct.dataclass
class Transition:
    # env
    obs: Any
    action: Any
    next_obs: Any
    reward: Any
    done: Any
    env_state: Any
    next_env_state: Any
    # policy
    policy_output: Any
    policy_state: Any = None
    next_policy_state: Any = None
    

class RolloutWrapper:
    def __init__(self, env: EnvBase, policy: Callable):
        self.env = env
        self.policy = policy
    
    def rollout(self, steps: int, env_param, policy_param, init, key):

        def rollout_step(step_input, _):
            (obs, env_state), key = step_input
            key, subkey = jax.random.split(key)
            action, policy_output = self.policy(obs, env_state, policy_param, subkey)
            next_obs, reward, done, next_state = self.env.step(env_state, action)
            output = Transition(obs, action, next_obs, reward, done, env_state, next_state, policy_output)
            next_obs, next_state = jax.lax.cond(
                done, 
                self.env.reset,
                lambda env_param, key: (next_obs, next_state),
                env_param, subkey
            )
            carry = ((next_obs, next_state), key)
            return carry, output
        
        if init is None:
            init = self.env.reset(env_param, key)

        (carry, _), output = jax.lax.scan(
            f=rollout_step,
            init=(init, key),
            xs=(),
            length=steps
        )
        return carry, output


class EnvTransform:
    def __init__(self) -> None:
        pass


class Compose(EnvTransform):
    def __init__(self) -> None:
        super().__init__()


class TransformedEnv(EnvBase):
    def __init__(self, env, transform: EnvTransform):
        pass
