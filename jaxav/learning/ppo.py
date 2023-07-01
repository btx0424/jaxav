import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax import struct
from jax.typing import ArrayLike

import optax
import distrax

from jaxav.base import Transition
from typing import Callable, Any, Tuple


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = getattr(nn, self.activation)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


@struct.dataclass
class PPOPolicyOutput:
    log_prob: ArrayLike
    value: ArrayLike


class PPOPolicy:
    def __init__(self):
        self.network = ActorCritic(4, activation="elu")
        self.gae = GAE(0.99, 0.95)
        self.clip_eps = 0.1
        self.num_minibatches = 8

    def init(self, x, key):
        tx = optax.chain(
            optax.clip_by_global_norm(10.),
            optax.adam(learning_rate=5e-3)
        )
        train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=self.network.init(key, x),
            tx=tx
        )
        return train_state

    def __call__(self, obs, env_state, params, key):
        pi, value = self.network.apply(params, obs)
        action, log_prob = pi.sample_and_log_prob(seed=key)
        return action, PPOPolicyOutput(log_prob, value)

    def update(self, traj_batch: Transition, train_state: TrainState, key):
        _, next_val = self.network.apply(train_state.params, traj_batch.next_obs[:, -1])
        advantages, returns = jax.vmap(self.gae)(traj_batch, next_val)

        batch = (traj_batch, advantages, returns)
        batch = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
        permutation = jax.random.permutation(key, batch[0].obs.shape[0])
        batch = jax.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
        batch = jax.tree_map(lambda x: x.reshape((self.num_minibatches, -1, *x.shape[1:])), batch)
        train_state, losses = jax.lax.scan(
            self._update_minibatch, train_state, batch
        )
        return train_state, jax.tree_map(jnp.mean, losses)
    
    def _update_minibatch(
        self, 
        train_state: TrainState, 
        minibatch: Tuple[Transition, jnp.ndarray, jnp.ndarray]
    ):
        transition, advantages, returns = minibatch

        def loss_fn(params, transition, advantages, returns):
            pi, value = self.network.apply(params, transition.obs)
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
            ratio = jnp.exp(log_prob - transition.policy_output.log_prob)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            loss_actor1 = ratio * advantages
            loss_actor2 = ratio.clip(1.-self.clip_eps, 1.+self.clip_eps) * advantages
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            total_loss = loss_actor + 0.5 * value_loss
            return total_loss, {"policy_loss": loss_actor, "value_loss": value_loss, "entropy": entropy}
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, losses), grads = grad_fn(
            train_state.params, transition, advantages, returns
        )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, losses


@struct.dataclass
class GAE:
    gamma: float
    lmbda: float

    def __call__(self, traj: Transition, next_val):
        def _get_advantages(gae_and_next_value, transition: Transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.policy_output.value,
                transition.reward,
            )
            delta = reward + self.gamma * next_value * (1 - done) - value
            gae = (
                delta
                + self.gamma * self.lmbda * (1 - done) * gae
            )
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(next_val), next_val),
            traj,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj.policy_output.value
