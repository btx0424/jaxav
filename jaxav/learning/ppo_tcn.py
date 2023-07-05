import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax import struct
from jax.typing import ArrayLike

import optax
from optax._src.linear_algebra import global_norm

import tensorflow_probability.substrates.jax.distributions as D

from jaxav.base import Transition
from jaxav.learning.common import GAE, ValueNorm, RunningStats
from typing import Callable, Any, Tuple


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "elu"

    @nn.compact
    def __call__(self, x):
        activation = getattr(nn, self.activation)
        actor_mean = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.Conv(features=64, kernel_size=[5], strides=2),
            activation,
            nn.Conv(features=64, kernel_size=[5], strides=2),
            activation,
            nn.Conv(features=64, kernel_size=[5], strides=2),
            lambda x: jnp.reshape(x, (*x.shape[:-2], -1)),
        ])(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = D.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        
        critic = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.Conv(features=64, kernel_size=[5], strides=2),
            activation,
            nn.Conv(features=64, kernel_size=[5], strides=2),
            activation,
            nn.Conv(features=64, kernel_size=[5], strides=2),
            lambda x: jnp.reshape(x, (*x.shape[:-2], -1)),
        ])(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return pi, critic


@struct.dataclass
class PPOPolicyOutput:
    log_prob: ArrayLike
    value: ArrayLike

class PPOTrainState(TrainState):
    value_stats: RunningStats

class PPOPolicyTCN:

    def __init__(self, cfg):
        self.cfg = cfg
        self.clip_eps = self.cfg.clip_eps
        self.num_minibatches = self.cfg.num_minibatches
        self.entropy_coef = self.cfg.entropy_coef

    def init(self, obs, key):
        self.network = ActorCritic(4, activation="elu")
        self.value_norm = ValueNorm()
        self.gae = GAE(0.99, 0.95)
        tx = optax.chain(
            optax.clip_by_global_norm(10.),
            optax.adam(learning_rate=self.cfg.lr)
        )
        def apply_fn(params, obs):
            return self.network.apply(params, jnp.swapaxes(obs, -2, -1))
        train_state = PPOTrainState.create(
            apply_fn=apply_fn,
            params=self.network.init(key, jnp.swapaxes(obs, -2, -1)),
            tx=tx,
            value_stats=RunningStats.zero()
        )
        return train_state

    def __call__(self, obs, env_state, params, key):
        obs = jnp.swapaxes(obs, -2, -1)
        pi, value = self.network.apply(params, obs)
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)
        return action, PPOPolicyOutput(log_prob, value)

    def update(self, traj_batch: Transition, train_state: PPOTrainState, key):
        _, next_val = train_state.apply_fn(
            train_state.params, traj_batch.next_obs[:, -1]
        )
        reward, value, done = (
            traj_batch.reward, 
            traj_batch.policy_output.value, 
            traj_batch.done
        )
        info = {}
        if self.value_norm is not None:
            value = self.value_norm.denormalize(train_state.value_stats, value)
            next_val = self.value_norm.denormalize(train_state.value_stats, next_val)
            advantages, returns = jax.vmap(self.gae)(reward, value, done, next_val)
            train_state = train_state.replace(
                value_stats=self.value_norm.update(train_state.value_stats, returns)
            )
            returns = self.value_norm.normalize(train_state.value_stats, returns)
            info["value_mean"] = train_state.value_stats.mean
        else:
            advantages, returns = jax.vmap(self.gae)(reward, value, done, next_val)

        batch = (traj_batch, advantages, returns)
        minibatches = self._make_minibatches(batch, key)
        train_state, _info = jax.lax.scan(
            self._update_minibatch, train_state, minibatches
        )
        info.update(_info)
        return train_state, jax.tree_map(jnp.mean, info)
    
    def _make_minibatches(self, batch, key):
        batch = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
        permutation = jax.random.permutation(key, batch[0].obs.shape[0])
        batch = jax.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
        batch = jax.tree_map(
            lambda x: x.reshape((self.num_minibatches, -1, *x.shape[1:])), batch
        )
        return batch

    def _update_minibatch(
        self, 
        train_state: TrainState, 
        minibatch: Tuple[Transition, jnp.ndarray, jnp.ndarray]
    ):
        transition, advantages, returns = minibatch

        def loss_fn(params, transition, advantages, returns):
            pi, value = train_state.apply_fn(params, transition.obs)
            log_prob = pi.log_prob(transition.action)

            # CALCULATE VALUE LOSS
            value_pred_clipped = transition.policy_output.value + (
                value - transition.policy_output.value
            ).clip(-self.clip_eps, self.clip_eps)
            value_losses = jnp.square(value - returns)
            value_losses_clipped = jnp.square(value_pred_clipped - returns)
            value_loss = (
                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            )

            # CALCULATE ACTOR LOSS
            ratio = jnp.exp(log_prob - transition.policy_output.log_prob)[..., None]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            loss_actor1 = ratio * advantages
            loss_actor2 = ratio.clip(1.-self.clip_eps, 1.+self.clip_eps) * advantages
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            total_loss = loss_actor + value_loss - self.entropy_coef * entropy
            explained_var = 1. - jnp.mean(value_losses) / jnp.var(returns)
            return (
                total_loss, 
                {
                    "policy_loss": loss_actor, 
                    "value_loss": value_loss, 
                    "entropy": entropy, 
                    "explained_var": explained_var
                }
            )
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, losses), grads = grad_fn(
            train_state.params, transition, advantages, returns
        )
        grad_norm = global_norm(grads)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, {"grad_norm": grad_norm, **losses}

