"""The module.
"""
from typing import List, Tuple
from needle.autograd import Tensor
from needle import ops
import needle.init as init

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        kwargs = dict(device=device, dtype=dtype, requires_grad=True)
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, **kwargs)
        )  # (in_features, out_features)
        self.bias = (
            Parameter(init.kaiming_uniform(out_features, 1, **kwargs).transpose())
            if bias
            else None
        )  # (1, out_features)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X @ self.weight
        if self.bias is not None:
            out += self.bias.broadcast_to(out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        B = X.shape[0]
        return X.reshape((B, X.size // B))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        B, N = logits.shape
        y_one_hot = init.one_hot(N, y)

        # sum 得到的标量  <class 'numpy.float32'> 而不是 NDArray 了
        # 而在 numpy 中  numpy.float32 / nums -->  numpy.float64
        # 因此 (ops.logsumexp(logits, (1,)).sum() - (y_one_hot * logits).sum()) / B 这样是错误的
        # 我们需要在成为标量前，就去除 scalar
        
        return (ops.logsumexp(logits, (1,)) / B).sum() - (y_one_hot * logits / B).sum()
        ### END YOUR SOLUTION


def norm_base(
    x: Tensor,
    axes: Tuple[int],
    weight: Parameter,
    bias: Parameter,
    eps: float,
    reshape_weight: bool = False,
):
    input_shape = x.shape
    new_shape = list(input_shape)
    t = 1
    for ax in axes:
        t *= input_shape[ax]
        new_shape[ax] = 1

    mean_ = x.sum(axes) / t
    mean = mean_.reshape(new_shape).broadcast_to(input_shape)

    var_ = ((x - mean) ** 2).sum(axes) / t
    var = var_.reshape(new_shape).broadcast_to(input_shape)
    std = (var + eps) ** 0.5

    if reshape_weight:
        weight = weight.reshape(new_shape)
        bias = bias.reshape(new_shape)

    out = weight.broadcast_to(input_shape) * (x - mean) / std + bias.broadcast_to(
        input_shape
    )
    return out, (mean_, var_)


# Applies Batch Normalization over a 2D or 3D input.
# MLP example: (N, C) dim = C   NLP example: (N, C, L) dim = C
# N is the batch size, C is the number of features or channels, and L is the sequence length
# Because the Batch Normalization is done over the C dimension, computing statistics on (N, L) slices, it’s common terminology to call this Temporal Batch Normalization.
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        kwargs = dict(device=device, dtype=dtype, requires_grad=True)
        self.weight = Parameter(init.ones(self.dim, **kwargs))
        self.bias = Parameter(init.zeros(self.dim, **kwargs))

        kwargs["requires_grad"] = False
        self.running_mean = init.zeros(self.dim, **kwargs)
        self.running_var = init.ones(self.dim, **kwargs)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape
        l = len(input_shape)
        assert l == 3 or l == 2, "BatchNorm1d only used for a 2D or 3D input."
        assert (
            self.dim == input_shape[1]
        ), f"number of channels or features of input_shape[1] {input_shape[1]} is not matched required dim {self.dim}"
        # 非 C computing statistics
        axes = (0,) if l == 2 else (0, 2)
        ### BEGIN YOUR SOLUTION
        if self.training:
            out, (mean, var) = norm_base(
                x, axes, self.weight, self.bias, self.eps, reshape_weight=(l == 3)
            )
            self.running_mean = (
                self.momentum * mean.data + (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * var.data + (1 - self.momentum) * self.running_var
            )
            return out
        else:
            mean = self.running_mean
            var = self.running_var
            weight = self.weight
            bias = self.bias

            if l == 3:
                new_shape = list(input_shape)
                for ax in axes:
                    new_shape[ax] = 1

                mean = mean.reshape(new_shape)
                var = var.reshape(new_shape)
                weight = weight.reshape(new_shape)
                bias = bias.reshape(new_shape)

            norm = (x - mean.broadcast_to(input_shape)) / (
                var.broadcast_to(input_shape) + self.eps
            ) ** 0.5
            return weight.broadcast_to(input_shape) * norm + bias.broadcast_to(
                input_shape
            )
        ### END YOUR SOLUTION


# MLP example: (N, C) dims = C   NLP Example: (N, L, C) dims = C   Image Example: (N, C, H, W) dims = [C, H, W]
# The mean and standard-deviation are calculated over the last D dimensions, where D is the length of dims
class LayerNorm(Module):
    def __init__(self, dims: List[int] | int, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dims = [dims] if type(dims) == int else dims
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        kwargs = dict(device=device, dtype=dtype, requires_grad=True)
        self.weight = Parameter(init.ones(*self.dims, **kwargs))
        self.bias = Parameter(init.zeros(*self.dims, **kwargs))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        input_shape = x.shape
        l = len(self.dims)
        assert (
            list(input_shape[-l:]) == self.dims
        ), f"input_shape {input_shape} is not matched required dims {self.dims}"
        axes = tuple([len(input_shape) - i - 1 for i in range(l)])
        out, _ = norm_base(x, axes, self.weight, self.bias, self.eps)
        return out
        ### END YOUR SOLUTION


# 可以推广到 任意维度的 LayerNorm
class LayerNorm1d(LayerNorm):
    def __init__(self, dim: int, eps=1e-5, device=None, dtype="float32"):
        super().__init__(dim, eps=eps, device=device, dtype=dtype)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p, device=x.device)
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
