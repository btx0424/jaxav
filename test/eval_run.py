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
from jaxav.envs import ENVS
from jaxav.learning.ppo import PPOPolicy

from omegaconf import DictConfig
import pprint
import logging


@hydra.main(config_path=CONFIG_PATH, config_name="eval", version_base=None)
def main(cfg: DictConfig):
    wandb.init(job_type="eval")
    artifact: wandb.Artifact = wandb.use_artifact(
        f"btx0424/jaxav/{cfg.env.name}-PPOPolicy:latest"
    )
    artifact_dir = artifact.download()
    atrifact_cfg = DictConfig(artifact.metadata)
    
    checkpoint_manager = CheckpointManager(
        artifact_dir,
        PyTreeCheckpointer(),
        CheckpointManagerOptions()
    )
    step = checkpoint_manager.latest_step()
    checkpoint = checkpoint_manager.restore(step)
    train_state = checkpoint["train_state"]

    drone = os.path.join(os.path.dirname(__file__), "../jaxav/asset/hummingbird.yaml")
    env = ENVS[cfg.env.name.lower()](drone)

    policy = PPOPolicy(atrifact_cfg.algo)
    collector = RolloutWrapperV0(env, policy, 500)

    @jax.jit
    def batch_rollout(env_params, policy_params, init, key):
        _batch_rollout = jax.vmap(collector.rollout, in_axes=(0, None, 0, 0))
        return _batch_rollout(env_params, policy_params, init, jax.random.split(key, num_envs))

    key = jax.random.PRNGKey(cfg.seed)
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
    from tqdm import tqdm
    traj_batch = to_numpy(batch.env_state)
    for i in tqdm(range(cfg.vis_traj_num)):
        traj = jax.tree_map(lambda x: x[i, ::2], traj_batch)
        traj = tree_unbind(traj)
        with open(f"eval_{i}.html", "w") as f:
            f.write(env.render_matplotlib(traj).to_jshtml())
    


if __name__ == "__main__":
    main()
