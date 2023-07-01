import jax
import jax.numpy as jnp
from flax import struct

from jaxav.dynamics import DroneState, step
from jaxav.pytree import tree_stack, to_numpy
from jaxav.visualization import render
from jaxav.control import RateController, PositionController
from jaxav.envs.hover import Hover
from jaxav.base import RolloutWrapper

import time

import yaml
drone = "jaxav/asset/hummingbird.yaml"
with open(drone, "r") as f:
    params = yaml.safe_load(f)

key = jax.random.PRNGKey(int(time.time()))

env = Hover(drone)

pos_ctrl = PositionController()
pos_ctrl_params = pos_ctrl.init(params)

@jax.jit
def policy(obs, env_state, ctrl_params):
    ctrl_ref = (
        jnp.zeros(7)
        .at[:3].set(jnp.array([0., 0., 2.5]))
    )
    action = pos_ctrl(env_state.drone, ctrl_ref, ctrl_params)
    return action

collector = RolloutWrapper(env, policy)
N = 2048
steps = 1024
start = time.perf_counter()
carry, output = collector.rollout(steps, None, pos_ctrl_params, None, key)
fps = steps / (time.perf_counter()-start)
print(fps)

start = time.perf_counter()
carry, output = collector.rollout(steps, None, pos_ctrl_params, None, key)
fps = steps / (time.perf_counter()-start)
print(fps)

start = time.perf_counter()
carry, output = jax.vmap(collector.rollout, (None, None, None, None, 0))(steps, None, pos_ctrl_params, None, jax.random.split(key, N))
fps = steps * N / (time.perf_counter()-start)
print(fps)