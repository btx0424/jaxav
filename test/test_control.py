import jax
import jax.numpy as jnp
from flax import struct

from jaxav.utils.pytree import to_numpy, tree_unbind
from jaxav.utils.math import quat_to_euler
from jaxav.control import PositionController
from jaxav.envs import Hover, Track
from jaxav.base import RolloutWrapperV0

import time
import os
import yaml
import matplotlib
import matplotlib.animation as animation
matplotlib.rcParams['animation.embed_limit'] = 2**128
writer = animation.FFMpegWriter(fps=30)

# drone = os.path.join(os.path.dirname(__file__), "../jaxav/asset/firefly.yaml")
drone = os.path.join(os.path.dirname(__file__), "../jaxav/asset/hummingbird.yaml")
with open(drone, "r") as f:
    params = yaml.safe_load(f)
pos_ctrl = PositionController()
pos_ctrl_params = pos_ctrl.init(params)

key = jax.random.PRNGKey(int(time.time()))
# key = jax.random.PRNGKey(0)

env = Hover(drone)

@jax.jit
def policy(obs, env_state, ctrl_params, key):
    # ctrl_ref = jnp.concatenate([
    #     jnp.array([0., 0., 2.5]),
    #     jnp.zeros(3),
    #     jnp.arctan2(env_state.ref_heading[1], env_state.ref_heading[0])[None]
    # ])
    ctrl_ref = jnp.concatenate([
        env_state.drone.pos,
        jnp.zeros(3),
        quat_to_euler(env_state.drone.rot)[2, None] + -0.3*jnp.pi
    ])
    action = pos_ctrl(env_state.drone, ctrl_ref, ctrl_params)
    return action, ()

collector = RolloutWrapperV0(env, policy, 1000)
env_param = env.init(key)
obs, env_state = env.reset(env_param, key)
init = (obs, env_state)
carry, output = collector.rollout(env_param, pos_ctrl_params, init, key)

traj = to_numpy(jax.tree_map(lambda x: x[::2], output.env_state))
traj = tree_unbind(traj)
anim = env.render_matplotlib(traj)
anim.save("test_control_hover.mp4", writer=writer)


# env = Track(drone)

# @jax.jit
# def policy(obs, env_state, ctrl_params, key):
#     ctrl_ref = jnp.concatenate([
#         env_state.ref_pos,
#         env_state.ref_vel,
#         # quat_to_euler(env_state.drone.rot)[2, None],
#         # jnp.array([0.])
#         jnp.arctan2(env_state.ref_heading[1], env_state.ref_heading[0])[None]
#     ])
#     action = pos_ctrl(env_state.drone, ctrl_ref, ctrl_params)
#     return action, ()

# collector = RolloutWrapperV0(env, policy, 1600)
# env_param = env.init(key)
# obs, env_state = env.reset(env_param, key)
# init = (obs, env_state)
# carry, output = collector.rollout(env_param, pos_ctrl_params, init, key)

# traj = to_numpy(jax.tree_map(lambda x: x[::2], output.env_state))
# traj = tree_unbind(traj)
# anim = env.render_matplotlib(traj)
# anim.save("test_control_track.mp4", writer=writer)

