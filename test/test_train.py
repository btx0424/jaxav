import jax
import jax.numpy as jnp

from jaxav import CONFIG_PATH
from jaxav.envs.hover import Hover
from jaxav.base import RolloutWrapper
from jaxav.learning.ppo import PPOPolicy
from jaxav.learning.ppo_rnn import PPOPolicyRNN
from jaxav.learning.ppo_tcn import PPOPolicyTCN
from jaxav.transform import (
    TransformedEnv,
    History
)

import time
import datetime
import yaml
import os
from tqdm import tqdm
import pprint
import wandb
import hydra

from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    run = wandb.init(
        project=cfg.wandb.project,
        name=f"Hover-{time_str}",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg)
    )

    drone = os.path.join(os.path.dirname(__file__), "../jaxav/asset/hummingbird.yaml")

    key = jax.random.PRNGKey(int(time.time()))

    # policy = PPOPolicy(cfg.algo)
    # policy = PPOPolicyRNN(cfg.algo)
    policy = PPOPolicyTCN(cfg.algo)

    base_env = Hover(drone)
    transform = History(32)
    env = TransformedEnv(base_env, transform)
    # env = base_env

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

    for i in tqdm(range(400)): 
        key, subkey = jax.random.split(key)
        carry, batch = batch_rollout(env_params, train_state.params, carry, subkey)
        train_state, info = train_step(batch, train_state, subkey)

        done = batch.done.all(-1)
        avg_return = batch.env_state.Return[done]
        metrics = jax.tree_map(lambda x: x[done], batch.env_state.metrics)
        info = {"info": info, "return": avg_return, "metrics": metrics}
        info = jax.tree_map(lambda x: jnp.mean(x).item(), info)

        pprint.pprint(info)
        run.log(info)

if __name__ == "__main__":
    main()