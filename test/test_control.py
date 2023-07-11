import jax
import jax.numpy as jnp
from flax import struct

from jaxav.dynamics import DroneState, step
from jaxav.utils.pytree import tree_stack, to_numpy, tree_split
from jaxav.utils.visualization import render
from jaxav.control import RateController, PositionController
from jaxav.envs.hover import Hover
from jaxav.base import RolloutWrapper
from jaxav.utils.visualization import render

import time
import os
import yaml

drone = os.path.join(os.path.dirname(__file__), "../jaxav/asset/hummingbird.yaml")
with open(drone, "r") as f:
    params = yaml.safe_load(f)

key = jax.random.PRNGKey(int(time.time()))

env = Hover(drone)

pos_ctrl = PositionController()
pos_ctrl_params = pos_ctrl.init(params)

@jax.jit
def policy(obs, env_state, ctrl_params, key):
    ctrl_ref = (
        jnp.zeros(7)
        .at[:3].set(jnp.array([0., 0., 2.5]))
    )
    action = pos_ctrl(env_state.drone, ctrl_ref, ctrl_params)
    return action, ()

steps = 500

collector = RolloutWrapper(env, policy, steps)
start = time.perf_counter()
env_param = env.init(key)
obs, env_state = env.reset(env_param, key)
init = (obs, env_state)
carry, output = collector.rollout(env_param, pos_ctrl_params, init, key)
fps = steps / (time.perf_counter()-start)
print(fps)

traj = output.env_state.drone
print(jax.tree_map(lambda x: x.shape, traj))
with open("test_control.html", "w") as f:
    f.write(render(tree_split(traj)[::2]).to_jshtml())

