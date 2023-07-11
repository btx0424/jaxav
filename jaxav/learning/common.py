import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Sequence

from flax import struct
from flax import linen as nn

from jaxav.utils.math import lerp

@struct.dataclass
class GAE:
    gamma: float
    lmbda: float

    def __call__(
        self, 
        reward: ArrayLike,
        value: ArrayLike,
        done: ArrayLike,
        next_val: ArrayLike
    ):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            reward, value, done = transition
            delta = reward + self.gamma * next_value * (1 - done) - value
            gae = (
                delta
                + self.gamma * self.lmbda * (1 - done) * gae
            )
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(next_val), next_val),
            (reward, value, done),
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + value


@struct.dataclass
class RunningStats:
    mean: ArrayLike
    sqr_mean: ArrayLike
    debias_term: ArrayLike

    @classmethod
    def zero(cls, shape=1):
        return cls(
            mean=jnp.zeros(shape),
            sqr_mean=jnp.zeros(shape),
            debias_term=jnp.array(1e-5)
        )


class ValueNorm:
    def __init__(self, beta: float=0.995):
        self.beta = jnp.array(beta)

    def update(self, stats: RunningStats, x: ArrayLike):
        x_mean = jnp.mean(x, axis=tuple(range(x.ndim-1)))
        x_sqr_mean = jnp.mean(jnp.square(x), axis=tuple(range(x.ndim-1)))
        new_stats = stats.replace(
            mean=lerp(stats.mean, x_mean, 1.-self.beta),
            sqr_mean=lerp(stats.sqr_mean, x_sqr_mean, 1.-self.beta),
            debias_term=lerp(stats.debias_term, 1., 1.-self.beta)
        )
        return new_stats
    
    def _mean_and_var(self, stats: RunningStats):
        mean = stats.mean / stats.debias_term
        sqr_mean = stats.sqr_mean / stats.debias_term
        var = jnp.clip(sqr_mean - jnp.square(mean), 1e-2)
        return mean, var

    def normalize(self, stats: RunningStats, x: ArrayLike):
        mean, var = self._mean_and_var(stats)
        return (x - mean) / jnp.sqrt(var)
    
    def denormalize(self, stats: RunningStats, x: ArrayLike):
        mean, var = self._mean_and_var(stats)
        return x * jnp.sqrt(var) + mean


class MLP(nn.Module):
    units: Sequence[int]
    activation: str = "elu"
    layer_norm: bool = False
    
    def setup(self) -> None:
        layers = []
        activation = getattr(nn, self.activation)
        for u in self.units:
            layers.append(nn.Dense(u))
            layers.append(activation)
            if self.layer_norm:
                layers.append(nn.LayerNorm())
        self.layers = nn.Sequential(layers)

    def __call__(self, x):
        return self.layers(x)