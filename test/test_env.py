from jaxav.envs.hover import Hover
from jaxav.transform import TransformedEnv, History
from jaxav.base import RolloutWrapper

import jax
import jax.numpy as jnp
import os

key = jax.random.PRNGKey(0)
drone = os.path.join(os.path.dirname(__file__), "../jaxav/asset/hummingbird.yaml")
env = Hover(drone)
env = TransformedEnv(env, History())
env_params = env.init(key)
obs, env_state = env.reset(env_params, key)

def policy(obs, env_state, _, key):
    print(env_state.info)
    return env.action_space.sample(key), {}

collector = RolloutWrapper(env, policy)
carry = (obs, env_state)
carry, output = collector.rollout(10, env_params, {}, carry, key)

info = output.env_state.info
def test(before, after):
    return jnp.all(after[1:] == before[:-1])

_, result = jax.lax.scan(
    f=lambda _, xs: ((), test(*xs)),
    init=(),
    xs=(info["obs_h"][:-1], info["obs_h"][1:])
)
print(result)