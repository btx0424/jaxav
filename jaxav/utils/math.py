import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Optional, Union, Tuple

def quat_inv(quat: ArrayLike):
    """Calculates the inverse of quaternion q.

    Args:
        q: (4,) quaternion [w, x, y, z]

    Returns:
        The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
    """
    return quat * jnp.array([1, -1, -1, -1])


def quat_mul(u: ArrayLike, v: ArrayLike) -> ArrayLike:
    """Multiplies two quaternions.

    Args:
        u: (4,) quaternion (w,x,y,z)
        v: (4,) quaternion (w,x,y,z)

    Returns:
        A quaternion u * v.
    """
    return jnp.array([
        u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
        u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
        u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
        u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
    ])
    

def quat_rotate(vec: ArrayLike, quat: ArrayLike, inv: bool=False):
    """Rotates a vector vec by a unit quaternion quat.

    Args:
        vec: (3,) a vector
        quat: (4,) a quaternion
        inv: bool whether to rotate by the inverted unit quaternion quat

    Returns:
        ndarray(3) containing vec rotated by quat.
    """
    if len(vec.shape) != 1:
        raise ValueError(f'vec must have no batch dimensions. vec.shape={vec.shape}')
    if inv:
        r = quat_rotate(vec, quat_inv(quat))
    else:
        s, u = quat[0], quat[1:]
        r = 2 * (jnp.dot(u, vec) * u) + (s * s - jnp.dot(u, u)) * vec
        r = r + 2 * s * jnp.cross(u, vec)
    return r


def ang_to_quat(ang: ArrayLike):
    """Converts angular velocity to a quaternion.

    Args:
        ang: (3,) angular velocity

    Returns:
        A rotation quaternion.
    """
    return jnp.array([0, ang[0], ang[1], ang[2]])


def quat_to_euler(quat: ArrayLike):
    w, x, y, z = quat
    euler_angles = jnp.array([
        jnp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
        jnp.arcsin(2.0 * (w * y - z * x)),
        jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
    ])
    return euler_angles


# def euler_to_quat(rpy: ArrayLike):
#     c1, c2, c3 = jnp.cos(rpy)
#     s1, s2, s3 = jnp.sin(rpy)
#     w = c1 * c2 * c3 - s1 * s2 * s3
#     x = s1 * c2 * c3 + c1 * s2 * s3
#     y = c1 * s2 * c3 - s1 * c2 * s3
#     z = c1 * c2 * s3 + s1 * s2 * c3
#     return jnp.array([w, x, y, z])


def euler_to_quat(euler: ArrayLike):
    r, p, y = euler
    cy = jnp.cos(y * 0.5)
    sy = jnp.sin(y * 0.5)
    cp = jnp.cos(p * 0.5)
    sp = jnp.sin(p * 0.5)
    cr = jnp.cos(r * 0.5)
    sr = jnp.sin(r * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quaternion = jnp.array([qw, qx, qy, qz])

    return quaternion

def quat_to_matrix(quat: ArrayLike):
    """Converts quaternion to 3x3 rotation matrix."""
    d = jnp.dot(quat, quat)
    w, x, y, z = quat
    s = 2 / d
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    return jnp.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])

# def quat_to_matrix(quat: ArrayLike):

#     w, x, y, z = quat
#     tx = 2.0 * x
#     ty = 2.0 * y
#     tz = 2.0 * z
#     twx = tx * w
#     twy = ty * w
#     twz = tz * w
#     txx = tx * x
#     txy = ty * x
#     txz = tz * x
#     tyy = ty * y
#     tyz = tz * y
#     tzz = tz * z

#     matrix = jnp.stack(
#         [
#             1 - (tyy + tzz),
#             txy - twz,
#             txz + twy,
#             txy + twz,
#             1 - (txx + tzz),
#             tyz - twx,
#             txz - twy,
#             tyz + twx,
#             1 - (txx + tyy),
#         ],
#         axis=-1,
#     ).reshape(3, 3)
#     return matrix

def safe_norm(
    x: ArrayLike, axis: Optional[Union[Tuple[int, ...], int]] = None
) -> ArrayLike:
    """Calculates a linalg.norm(x) that's safe for gradients at x=0.

    Avoids a poorly defined gradient for jnp.linal.norm(0) see
    https://github.com/google/jax/issues/3058 for details
    Args:
        x: A jnp.array
        axis: The axis along which to compute the norm

    Returns:
        Norm of the array x.
    """

    is_zero = jnp.allclose(x, 0.0)
    # temporarily swap x with ones if is_zero, then swap back
    x = jnp.where(is_zero, jnp.ones_like(x), x)
    n = jnp.linalg.norm(x, axis=axis)
    n = jnp.where(is_zero, 0.0, n)
    return n

# def normalize(
#     x: ArrayLike, axis: Optional[Union[Tuple[int, ...], int]] = None
# ) -> Tuple[ArrayLike, ArrayLike]:
#     """Normalizes an array.

#     Args:
#         x: A jnp.array
#         axis: The axis along which to compute the norm

#     Returns:
#         A tuple of (normalized array x, the norm).
#     """
#     norm = safe_norm(x, axis=axis)
#     n = x / (norm + 1e-6 * (norm == 0.0))
#     return n, norm

def normalize(x: ArrayLike):
    norm = jnp.linalg.norm(x, axis=-1)
    return x / (norm + 1e-6), norm


def lerp(start: ArrayLike, end: ArrayLike, weight: Union[ArrayLike, float]):
    return start + (end - start) * weight


def uniform(key: jax.random.KeyArray, low: ArrayLike, high: ArrayLike):
    return jax.random.uniform(key) * (high-low) + low

def sign(key: jax.random.KeyArray):
    return jnp.sign(jax.random.normal(key))

