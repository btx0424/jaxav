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

from typing import Callable, Any, Tuple

from jaxav.base import Transition
from jaxav.learning.common import GAE
from jaxav.learning.ppo import PPOPolicyOutput


class ScannedRNN(nn.Module):
    
    @nn.compact
    def __call__(self, rnn_state, x, is_init):
        rnn_state = jnp.where(
            is_init,
            jnp.zeros_like(rnn_state),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell()(rnn_state, x)
        return new_rnn_state, y


class ActorCriticRNN(nn.Module):
    action_dim: int
    activation: str = "elu"

    @nn.compact
    def __call__(self, rnn_state, obs, is_init):
        activation = getattr(nn, self.activation)

        embedding = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = activation(embedding)

        rnn_state, embedding = ScannedRNN()(rnn_state, embedding, is_init)

        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = D.MultivariateNormalDiag(
            actor_mean, jnp.broadcast_to(jnp.exp(actor_logtstd), actor_mean.shape)
        )

        critic = nn.Dense(
            256, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return rnn_state, (pi, jnp.squeeze(critic, axis=-1))


class PPOPolicyRNN:

    def __init__(self):
        self.network = ActorCriticRNN(4, activation="elu")
        self.gae = GAE(0.99, 0.95)
        self.clip_eps = 0.1
        self.num_minibatches = 8
        self.seq_len = 64

    def init(self, obs, key):
        tx = optax.chain(
            optax.clip_by_global_norm(10.),
            optax.adam(learning_rate=5e-4)
        )
        rnn_state = nn.GRUCell.initialize_carry(key, (obs.shape[0],), 128)
        is_init = jnp.ones((obs.shape[0], 1), bool)
        apply_fn = nn.scan(
            ActorCriticRNN,
            in_axes=1,
            out_axes=1,
            variable_broadcast="params",
            split_rngs={"params": False},
        )(4, activation="elu")
        train_state = TrainState.create(
            apply_fn=apply_fn.apply,
            params=self.network.init(key, rnn_state, obs, is_init),
            tx=tx
        )
        return train_state

    def __call__(self, obs, env_state, policy_state, params, key):
        rnn_state, (pi, value) = self.network.apply(
            params, policy_state, obs, env_state.is_init, 
        )
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)
        return action, PPOPolicyOutput(log_prob, value), rnn_state
    
    def reset(self, key):
        rnn_state = nn.GRUCell.initialize_carry(key, (), 128)
        return rnn_state

    def update(self, traj_batch: Transition, train_state: TrainState, key):
        _, (_, next_val) = self.network.apply(
            train_state.params, 
            traj_batch.policy_state[:, -1],
            traj_batch.next_obs[:, -1],
            traj_batch.next_env_state.is_init[:, [-1]]
        )
        advantages, returns = jax.vmap(self.gae)(traj_batch, next_val)

        batch = (traj_batch, advantages, returns)
        minibatches = self._make_minibatches(batch, key)
        train_state, losses = jax.lax.scan(
            self._update_minibatch, train_state, minibatches
        )
        return train_state, jax.tree_map(jnp.mean, losses)
    
    def _make_minibatches(self, batch: Transition, key):
        batch = jax.tree_map(
            lambda x: x.reshape(-1, self.seq_len, *x.shape[2:]), batch
        )
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
            rnn_state = transition.policy_state[:, 0]
            is_init = transition.env_state.is_init[..., None]
            _, (pi, value) = train_state.apply_fn(
                params, rnn_state, transition.obs, is_init
            )
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
        grad_norm = global_norm(grads)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, {"grad_norm": grad_norm, **losses}
