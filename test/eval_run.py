import wandb
import hydra
import os
import jax
import jax.numpy as jnp

from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer
)
from flax.training.train_state import TrainState

from jaxav import CONFIG_PATH
from jaxav.base import RolloutWrapperV0
from jaxav.envs import Hover, Track
from jaxav.learning.ppo import PPOPolicy

from omegaconf import DictConfig
import pprint
import logging


# @hydra.main(config_path=CONFIG_PATH, config_name="eval", version_base=None)
def main():
    wandb.init()
    artifact: wandb.Artifact = wandb.use_artifact("btx0424/jaxav/Track-PPOPolicy:v0")
    artifact_dir = artifact.download()
    cfg = DictConfig(artifact.metadata)
    
    checkpoint_manager = CheckpointManager(
        artifact_dir,
        PyTreeCheckpointer(),
        CheckpointManagerOptions()
    )
    step = checkpoint_manager.latest_step()
    checkpoint = checkpoint_manager.restore(step)
    train_state = checkpoint["train_state"]

    drone = os.path.join(os.path.dirname(__file__), "../jaxav/asset/hummingbird.yaml")
    env = Track(drone)
    # env = Hover(drone)
    policy = PPOPolicy(cfg.algo)
    collector = RolloutWrapperV0(env, policy, 800)

    @jax.jit
    def batch_rollout(env_params, policy_params, init, key):
        _batch_rollout = jax.vmap(collector.rollout, in_axes=(0, None, 0, 0))
        return _batch_rollout(env_params, policy_params, init, jax.random.split(key, num_envs))

    key = jax.random.PRNGKey(0)
    num_envs = cfg.env.num_envs
    env_params = jax.vmap(env.init)(jax.random.split(key, num_envs))
    carry = jax.vmap(collector.init)(env_params, jax.random.split(key, num_envs))
    
    carry, batch = batch_rollout(env_params, train_state["params"], carry, key)
    done = batch.done.all(-1)
    avg_return = batch.env_state.Return[done]
    metrics = jax.tree_map(lambda x: x[done], batch.env_state.metrics)
    
    info = {"return": avg_return, "metrics": metrics}
    info = jax.tree_map(lambda x: jnp.mean(x).item(), info)
    
    pprint.pprint(info)
    
    from jaxav.utils.pytree import tree_unbind, to_numpy
    import time
    print(time.perf_counter())
    traj = jax.tree_map(lambda x: x[1, ::2], to_numpy(batch.env_state))
    traj = tree_unbind(traj)
    print(time.perf_counter())
    with open("eval.html", "w") as f:
        f.write(env.render_matplotlib(traj).to_jshtml())
    


if __name__ == "__main__":
    main()
