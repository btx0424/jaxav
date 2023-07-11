import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from typing import List, TypeVar

T = TypeVar("T")

def tree_stack(trees, axis=0):
    return jax.tree_map(lambda *args: jnp.stack(args, axis), *trees)


def to_numpy(tree):
    return jax.tree_map(np.array, tree)


def tree_unbind(tree: T, axis: int=0) -> List[T]:
    flat, treedef = tree_util.tree_flatten(tree)
    unbinded_flat = jax.tree_map(lambda x: np.split(x, x.shape[axis], axis), flat)
    unbinded_flat = jax.tree_map(lambda x: np.squeeze(x, axis), unbinded_flat)
    unbinded_tree = list(
        map(lambda x: tree_util.tree_unflatten(treedef, x), zip(*unbinded_flat))
    )
    return unbinded_tree
    