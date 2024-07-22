"""Operator implementations."""

from typing import Optional

from ..autograd import NDArray, TensorTuple, TensorTupleOp
from ..autograd import Tensor, TensorOp
from .ops_tuple import make_tuple

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
# import numpy
# import numpy as array_api

from .. import backend_ndarray as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * self.scalar * (a ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, -a * out_grad / b**2
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # 1. 只考虑交换两个维度
        # 2. array_api.permute 是重新排列轴，例如 (0,1) 是不会交换的，必须要 (1,0) 才会交换
        # array_api.swapaxes 无论是 (0,1) 还是 (1,0) 都会交换这两个轴
        if self.axes is not None:
            return a.swapaxes(self.axes[0], self.axes[1])
        else:
            return a.swapaxes(a.ndim - 2, a.ndim - 1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(
            self.axes
        )  # transpose back, op return a new value, not in-place change
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(
            node.inputs[0].shape
        )  # back to input shape, gradient shape same as input shape
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.broadcast_to(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 在 broadcast 维度上求和，因为 broadcast 节点本质上就是
        # 一个输入，然后通过复用到多个输出（注意不是拷贝）
        input_shape = node.inputs[0].shape
        sum_axes = list(range(len(self.shape)))
        # broadcast 是不改变 ndim
        """
        但需要注意一个特殊情况 (n,) 可以 broadcast_to (m,n)
        这个时候需要返过来遍历
        """
        for i, (ori, cur) in enumerate(
            zip(reversed(input_shape), reversed(self.shape))
        ):
            if cur == ori:
                sum_axes[len(self.shape) - i - 1] = -1  # 没有 broadcast 的轴
        sum_axes = tuple(filter(lambda x: x >= 0, sum_axes))
        return out_grad.sum(sum_axes).reshape(input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = (axes,) if isinstance(axes, int) else axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        new_shape = list(input_shape)
        sum_axes = range(len(new_shape)) if self.axes is None else self.axes
        for axes in sum_axes:
            new_shape[axes] = 1  # 对 sum 轴填充 1
        return out_grad.reshape(new_shape).broadcast_to(input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        a_grad, b_grad = out_grad @ b.transpose(), a.transpose() @ out_grad
        # 如果梯度的维数比原始形状多，沿着额外的轴求和
        # (6,6,5,4) @ (4,3) 你会发现，此时 b_grad (6,6,4,3)
        if len(a.shape) < len(a_grad.shape):
            a_grad = a_grad.sum(
                tuple([i for i in range(len(a_grad.shape) - len(a.shape))])
            )
        if len(b.shape) < len(b_grad.shape):
            b_grad = b_grad.sum(
                tuple([i for i in range(len(b_grad.shape) - len(b.shape))])
            )
        return a_grad, b_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.log()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - tanh(node.inputs[0]) ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.maximum(0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 尽可能拷贝，不要自己新建，很容易忘记设置 dtype 和 device
        # 注意这里是 node，不是 node.input，node.input 有小于 0 的，而 relu 在小于 0 处的梯度都是 0
        out = node.realize_cached_data()
        return out_grad * Tensor(out > 0, device=out_grad.device, dtype=out_grad.dtype)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(args)>=1, "stack must has at least one array!"
        shape = args[0].shape
        device = args[0].device
        dtype = args[0].dtype
        for a in args:
            assert a.shape == shape, "All arrays need to be of the same shape!"
        new_shape = list(shape)
        new_shape.insert(self.axis, len(args))  # new dimension is len(args)
        out = array_api.empty(new_shape, device=device, dtype=dtype)
        slices = [slice(0,s) for s in new_shape] # 每个轴完整的 slice
        for i, arr in enumerate(args):
            slices[self.axis] = slice(i, i+1)  # stack 的轴的 slice 为索引 i  --> [:,:,i,:,:]
            out[tuple(slices)] = arr
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        shape = A.shape
        n = shape[self.axis]
        new_shape = list(shape)
        new_shape.pop(self.axis)

        slices = [slice(0,s) for s in shape]
        splits = []
        for i in range(n):
            slices[self.axis] = slice(i, i+1)
            splits.append(A[tuple(slices)].reshape(new_shape)) # 直接截取出来即可

        return tuple(splits)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape) 
        for ax in self.axes:
            new_shape[ax] *= (self.dilation + 1)
        out = array_api.empty(tuple(new_shape), dtype=a.dtype, device=a.device)
        out.fill(0)
        slices = [slice(0,s) for s in new_shape]
        for ax in self.axes:
            slices[ax] = slice(0, new_shape[ax], self.dilation + 1)
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = a.shape
        slices = [slice(0, s) for s in new_shape]
        for ax in self.axes:
            slices[ax] = slice(0, new_shape[ax], self.dilation + 1)
        return a[tuple(slices)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A: NDArray, B: NDArray):
        '''
        A is input image NHWC
        B is kernel KKCC_o
        '''
        ### BEGIN YOUR SOLUTION
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N,H,W,C_in = A.shape
        K_H,K_W,C_in_,C_out = B.shape
        Ns,Hs,Ws,Cs = A.strides
        stride = self.stride
        assert K_H == K_W, "Conv kernel should be a square tensor"
        assert C_in == C_in_, "Conv kernel and input are not compatible"

        out_H, out_W = (H - K_H + 1) // stride, (W - K_W + 1) // stride
        inner_size = K_H*K_W*C_in_

        img2col = A.as_strided(
            (N, out_H, out_W, K_H, K_W, C_in), (Ns, Hs * stride, Ws * stride, Hs, Ws, Cs)
        ).reshape((N * out_H * out_W, inner_size))

        out = img2col @ B.reshape((inner_size, C_out))

        return out.reshape((N, out_H, out_W, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        """
        out_grad is N, out_H, out_W, C_o
        node.inputs[0] is input image NHWC
        node.inputs[1] is kernel KKCC_o
        """
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        K = W.shape[0]
        if self.stride > 1:
            out_grad = dilate(
                out_grad, (1, 2), self.stride - 1
            )  # NH_oW_oC_o --> N(H+2P-K+1)(W+2P-K+1)C_o 抵消步长

        # X_grad
        W_flipped = transpose(flip(W, (0, 1)), (2, 3))  # KKCC_o -->  KKC_oC
        X_grad = conv(
            out_grad, W_flipped, padding=K - 1 - self.padding
        )  # N(H+2P-K+1)(W+2P-K+1)C_o  卷积   KKC_oC  --> NHWC  stride=1 padding=K-1-P
        # H+2P-K+1+2(K-1-P)-K+1 卷积后刚好等于 H

        # W_grad 的计算以 N 为 C_in 进行卷积
        grad_permute = transpose(
            transpose(out_grad, (0, 1)), (1, 2)
        )  # N(H+2P-K+1)(W+2P-K+1)C_o --> (H+2P-K+1)(W+2P-K+1)NC_o 为卷积核
        X_permute = transpose(X, (0, 3))  # NHWC  --> CHWN  输入图像
        W_grad = conv(X_permute, grad_permute, padding=self.padding)  #  CHWN  卷积  (H+2P-K+1)(W+2P-K+1)NC_o  --> CKKC_o，那么 padding 需要保持一致
        # H+2P-(H+2P-K+1)+1  卷积后刚好等于  K
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2))  # CKKC_o  --> KKCC_o

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
