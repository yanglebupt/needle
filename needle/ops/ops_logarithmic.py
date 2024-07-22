from typing import Optional
from ..autograd import Tensor, TensorOp
from .ops_mathematic import *

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = (axes,) if isinstance(axes, int) else axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, self.axes, keepdims=True)
        out = array_api.log( array_api.exp(Z - max_z.broadcast_to(Z.shape)).sum(self.axes) )
        return out + max_z.reshape(out.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 梯度可以拆成三部分
        Z = node.inputs[0]
        max_z = Tensor(array_api.max(Z.realize_cached_data(), self.axes, keepdims=True), device=Z.device, dtype=Z.dtype)
        exp_z = exp(Z - max_z.broadcast_to(Z.shape))
        sum_exp_z = exp_z.sum(self.axes)
        grad = out_grad / sum_exp_z

        input_shape = Z.shape
        new_shape = list(input_shape)
        sum_axes = list(range(len(new_shape))) if self.axes is None else self.axes
        for axes in sum_axes:
            new_shape[axes] = 1

        grad = grad.reshape(new_shape).broadcast_to(input_shape)  
        return grad * exp_z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
