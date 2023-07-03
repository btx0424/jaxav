import jax
import jax.numpy as jnp

from jaxav.envs.hover import Hover
from jaxav.base import RolloutWrapper
from jaxav.learning.ppo import PPOPolicy
from jaxav.learning.ppo_rnn import PPOPolicyRNN

import time
import yaml
import os
from tqdm import tqdm
import pprint

def main():
    drone = os.path.join(os.path.dirname(__file__), "../jaxav/asset/hummingbird.yaml")

    key = jax.random.PRNGKey(int(time.time()))

    env = Hover(drone)
    # policy = PPOPolicy()
    policy = PPOPolicyRNN()

    num_envs = 1024
    steps = 128

    env_params = jax.vmap(env.init)(jax.random.split(key, num_envs))

    collector = RolloutWrapper(env, policy)

    @jax.jit
    def batch_rollout(env_params, policy_params, init, key):
        _batch_rollout = jax.vmap(collector.rollout, in_axes=(None, 0, None, 0, 0))
        return _batch_rollout(steps, env_params, policy_params, init, jax.random.split(key, num_envs))

    obs, env_state = jax.vmap(env.reset)(env_params, jax.random.split(key, num_envs))
    if hasattr(policy, "reset"): 
        # stateful poicy, e.g., RNN, PID controller
        train_state = policy.init(obs, key)
        policy_state = jax.vmap(policy.reset)(jax.random.split(key, num_envs))
        carry = (obs, env_state, policy_state)
    else:
        # stateless policy
        carry = (obs, env_state)
        train_state = policy.init(obs, key)

    train_step = jax.jit(policy.update)

    for i in tqdm(range(300)): 
        key, subkey = jax.random.split(key)
        carry, batch = batch_rollout(env_params, train_state.params, carry, subkey)
        train_state, info = train_step(batch, train_state, subkey)

        avg_return = batch.env_state.Return[batch.done]
        metrics = jax.tree_map(lambda x: x[batch.done], batch.env_state.metrics)
        info = {"info": info, "return": avg_return, "metrics": metrics}

        pprint.pprint(jax.tree_map(lambda x: jnp.mean(x).item(), info))

if __name__ == "__main__":
    main()