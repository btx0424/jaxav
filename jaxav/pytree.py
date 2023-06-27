import jax
import jax.numpy as jnp
import numpy as np

def tree_stack(trees, axis=0):
    return jax.tree_map(lambda *args: jnp.stack(args, axis), *trees)

def to_numpy(tree):
    return jax.tree_map(np.array, tree)
