import numpy as np
from .ndarray import NDArray

def copy(a: NDArray):
    return a.copy()


def broadcast_to(a: NDArray, new_shape):
    return a.broadcast_to(new_shape)


def sum(a: NDArray, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)


def max(a: NDArray, axis=None, keepdims=False):
    return a.max(axis=axis, keepdims=keepdims)

def maximum(a: NDArray, b:NDArray):
    return a.maximum(b)

def log(a: NDArray):
    return a.log()


def exp(a: NDArray):
    return a.exp()


def tanh(a: NDArray):
    return a.tanh()


### Convenience methods to match numpy a bit more closely
def array(a, dtype=None, device=None):
    return NDArray(a, dtype=dtype, device=device)


def empty(shape, dtype=None, device=None):
    return NDArray.make(shape, dtype=dtype, device=device)


def full(shape, fill_value, dtype=None, device=None):
    arr = empty(shape, dtype, device)
    arr.fill(fill_value)
    return arr


def zeros(shape, dtype=None, device=None):
    return full(shape, 0, dtype, device)


def ones(shape, dtype=None, device=None):
    return full(shape, 1, dtype, device)


def randn(shape, dtype=None, device=None):
    # note: numpy doesn't support types within standard random routines, and
    # .astype("float32") does work if we're generating a singleton
    return NDArray(np.random.randn(*shape).astype(dtype), dtype=dtype, device=device)


def rand(shape, dtype=None, device=None):
    # note: numpy doesn't support types within standard random routines, and
    # .astype("float32") does work if we're generating a singleton
    return NDArray(np.random.rand(*shape).astype(dtype), dtype=dtype, device=device)


def one_hot(n, i, dtype=None, device=None):
    # (n,n) 的单位矩阵，然后根据 y 的值（类别索引）取行，就可以得到 one-hot 的矩阵
    return NDArray(np.eye(n, dtype=dtype)[i.astype(np.int32)], dtype=dtype, device=device)
