import jax
import jax.numpy as jnp

from flax import struct
from typing import Callable, Any, Dict
from dataclasses import field

@struct.dataclass
class EnvState:
    max_episode_len: int
    
    dt: float = field(default=0.02, kw_only=True)
    is_init: bool = field(default=True, kw_only=True)
    step: int = field(default=0, kw_only=True)
    Return: float = field(default=jnp.array([0.]), kw_only=True)
    
    metrics: Dict[str, Any] = field(default_factory=dict, kw_only=True)
    info: Dict[str, Any] = field(default_factory=dict, kw_only=True)


class EnvBase:
    def __init__(self) -> None:
        self.MAX_EPISODE_LEN: int
    
    def init(self, key: jax.random.KeyArray):
        pass

    def step(self, state: EnvState, action):
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


class RolloutWrapperV0:
    def __init__(
        self, 
        env: EnvBase, 
        policy: Callable, 
        steps_per_iter: int
    ):
        self.env = env
        self.policy = policy
        self.steps_per_iter = steps_per_iter
    
    def init(self, env_params, key):
        obs, env_state = self.env.reset(env_params, key)
        if hasattr(self.policy, "reset"): 
            # stateful poicy, e.g., RNN, PID controller
            policy_state = self.policy.reset(key)
            carry = (obs, env_state, policy_state)
        else:
            # stateless policy
            carry = (obs, env_state)
        return carry
        
    def rollout(self, env_param, policy_param, init, key):
        if len(init) == 2:
            def rollout_step(step_input, _):
                (obs, env_state), key = step_input
                key, subkey = jax.random.split(key)
                action, policy_output = self.policy(obs, env_state, policy_param, subkey)
                next_obs, reward, done, next_state = self.env.step(env_state, action)
                output = Transition(obs, action, next_obs, reward, done, env_state, next_state, policy_output)
                next_obs, next_state = jax.lax.cond(
                    done.all(-1), 
                    self.env.reset,
                    lambda env_param, key: (next_obs, next_state),
                    env_param, subkey
                )
                carry = ((next_obs, next_state), key)
                return carry, output
        elif len(init) == 3:
            def rollout_step(step_input, _):
                (obs, env_state, policy_state), key = step_input
                key, subkey = jax.random.split(key)
                action, policy_output, policy_state = self.policy(
                    obs, env_state, policy_state, policy_param, subkey
                )
                next_obs, reward, done, next_state = self.env.step(env_state, action)
                output = Transition(
                    obs, action, next_obs, reward, done, env_state, next_state, 
                    policy_output=policy_output,
                    policy_state=policy_state
                )
                next_obs, next_state = jax.lax.cond(
                    done.all(-1), 
                    self.env.reset,
                    lambda env_param, key: (next_obs, next_state),
                    env_param, subkey
                )
                carry = ((next_obs, next_state, policy_state), key)
                return carry, output
        else:
            raise ValueError
        (carry, _), output = jax.lax.scan(
            f=rollout_step,
            init=(init, key),
            xs=(),
            length=self.steps_per_iter
        )
        return carry, output

