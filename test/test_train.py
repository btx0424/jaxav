import jax
import jax.numpy as jnp

from jaxav import CONFIG_PATH
from jaxav.envs import ENVS
from jaxav.base import RolloutWrapperV0
from jaxav.learning.ppo import PPOPolicy
from jaxav.learning.ppo_rnn import PPOPolicyRNN
from jaxav.learning.ppo_tcn import PPOPolicyTCN
from jaxav.transform import (
    TransformedEnv,
    History
)
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer
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


@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    run = wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.env.name}-{time_str}",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg)
    )
    pprint.pprint(OmegaConf.to_container(cfg))

    drone = os.path.join(os.path.dirname(__file__), "../jaxav/asset/hummingbird.yaml")

    key = jax.random.PRNGKey(int(time.time()))

    policy = PPOPolicy(cfg.algo)
    # policy = PPOPolicyRNN(cfg.algo)
    # policy = PPOPolicyTCN(cfg.algo)

    base_env = ENVS[cfg.env.name.lower()](drone)
    # transform = History(32)
    # env = TransformedEnv(base_env, transform)
    env = base_env

    num_envs = cfg.env.num_envs
    steps = 128

    collector = RolloutWrapperV0(env, policy, steps)
    checkpoint_manager = CheckpointManager(
        os.path.join(run.dir, "checkpoints"),
        PyTreeCheckpointer(),
        CheckpointManagerOptions()
    )

    @jax.jit
    def batch_rollout(env_params, policy_params, init, key):
        _batch_rollout = jax.vmap(collector.rollout, in_axes=(0, None, 0, 0))
        return _batch_rollout(env_params, policy_params, init, jax.random.split(key, num_envs))

    env_params = jax.vmap(env.init)(jax.random.split(key, num_envs))
    carry = jax.vmap(collector.init)(env_params, jax.random.split(key, num_envs))
    train_state = policy.init(carry[0], key)

    train_step = jax.jit(policy.update)

    for i in tqdm(range(400)): 
        key, subkey = jax.random.split(key)
        iter_start = time.perf_counter()
        carry, batch = batch_rollout(env_params, train_state.params, carry, subkey)
        rollout_end = time.perf_counter()
        train_state, info = train_step(batch, train_state, subkey)
        train_end = time.perf_counter()

        done = batch.done.all(-1)
        avg_return = batch.env_state.Return[done]
        metrics = jax.tree_map(lambda x: x[done], batch.env_state.metrics)
        info = {
            "info": info, 
            "return": avg_return, 
            "metrics": metrics, 
            "rollout_time": rollout_end-iter_start,
            "training_time": train_end-iter_start
        }
        info = jax.tree_map(lambda x: jnp.mean(x).item(), info)

        pprint.pprint(info)
        run.log(info)

    checkpoint_manager.save(400, {"train_state": train_state})
    
    if not run.disabled:
        model_artifact = wandb.Artifact(
            name=f"{env.__class__.__name__}-{policy.__class__.__name__}", 
            type="train_state",
            metadata=OmegaConf.to_container(cfg)
        )
        model_artifact.add_dir(checkpoint_manager.directory)
        run.log_artifact(model_artifact, aliases=["latest", run.path.replace("/", "-")])


if __name__ == "__main__":
    main()
