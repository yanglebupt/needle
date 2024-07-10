from functools import reduce
from operator import mul
from typing import Any, Optional, Tuple, Union
import numpy as np
from .device import BackendDevice, default_device
from .ndtype import ndtype, default_dtype, float32, float64

def prod(x):
    return reduce(mul, x, 1)

class NDArray:
    dtype: ndtype
    device: BackendDevice

    shape: Tuple[int]
    strides: Tuple[int]
    offset: int
    # 存贮的 cpp 指针类
    _handle: Any  

    def __init__(
        self,
        array: Union["NDArray", np.ndarray, list],
        dtype: Optional[ndtype] = None,
        device: Optional[BackendDevice] = None,
    ):
        """array is python nested list, it will be transfered to c++"""
        if isinstance(array, NDArray):
            new_array = array + 0.0  # copy
            NDArray.make(
                new_array.shape,
                self = self,
                device = new_array.device,
                dtype = new_array.dtype if dtype is None else dtype,
                strides = new_array.strides,
                offset = new_array.offset,
                handle = new_array._handle,
            )
            # 是否重新指定 device
            if device is not None and device != new_array.device:
                self.to(device)
        elif isinstance(array, np.ndarray):
            NDArray.make(
                array.shape,
                self = self,
                device = device,
                dtype = array.dtype if dtype is None else dtype,
            )
            # copy array to cpp ptr
            self.device.from_array(np.ascontiguousarray(array), self._handle)
        else:
            self.dtype = default_dtype if dtype is None else dtype
            # you can use flatten_get_shape_of_list to get shape for nested list
            # but you can also use np.array for simply
            flatten, shape = NDArray.flatten_get_shape_of_list(array)
            NDArray.make(
                shape,
                self = self,
                device = device,
                dtype = dtype,
            )
            # copy array to cpp ptr
            self.device.from_array((self.dtype.ctype*len(flatten))(*flatten), self._handle)

    @staticmethod
    def map_dtype(dtype):
        """change numpy dtype to my NDArray dtype"""
        if isinstance(dtype, ndtype):
            return dtype
        if dtype == np.float32 or dtype == "float32":
            return float32
        elif dtype == np.float64 or dtype == "float64":
            return float64
        else:
            # raise Exception(f"only support np.float32 and np.float64, but input dtype is {dtype}")
            return float32

    # can use numpy.array replace this method
    @staticmethod
    def flatten_get_shape_of_list(nested_list):
        """
        This method simply convert a list type tensor to a flatten tensor with its shape

        Example:

        Arguments:
            nested_list: [[1, 2, 3], [-5, 2, 0]]
        Return:
            flat_data: [1, 2, 3, -5, 2, 0]
            shape: [2, 3]
        """
        flat_data = []
        shape = []
        if isinstance(nested_list, list):
            for sublist in nested_list:
                inner_data, inner_shape = NDArray.flatten_get_shape_of_list(sublist)
                flat_data.extend(inner_data)  # data 拼接起来
            # 内部元素遍历完后，当前 shape 拼接 内部 shape（递归归并的思路）
            shape.append(len(nested_list))
            shape.extend(inner_shape)
        else:  # 递归终止条件，添加单个数，shape 为空
            flat_data.append(nested_list)
        return flat_data, shape

    @staticmethod
    def compact_strides(shape):
        """get row major strides"""
        strides = [0] * (n:=len(shape))
        stride = 1
        for i in range(n-1, -1, -1):
            strides[i] = stride
            stride *= shape[i]
        return tuple(strides)

    @staticmethod
    def make(shape, self=None, device=None, dtype=None, strides=None, offset=0, handle=None):
        """create or initialize a NDArray with all necessary attributes"""
        array = NDArray.__new__(NDArray) if self is None else self
        array.shape = tuple(shape)
        array.strides = NDArray.compact_strides(shape) if strides is None else strides
        array.device = default_device() if device is None else device
        array.dtype = default_dtype if dtype is None else  NDArray.map_dtype(dtype)
        array.offset = offset
        # cpp 中开辟一块对齐的内存地址
        array._handle = array.device.Array(array.size) if handle is None else handle
        return array

    @property
    def size(self):
        return prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return f"NDArray({str(self.numpy())}, dtype={self.dtype}, device={self.device})"

    def __str__(self):
        return str(self.numpy())

    def numpy(self):
        """convert to a numpy array"""
        return self.device.to_numpy(
            self._handle, list(self.shape), list(self.strides), self.offset
        )

    ### Basic array manipulation
    def fill(self, value):
        self.device.fill(self._handle, value); 

    def to(self, device):
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device, dtype=self.dtype)

    def is_compact(self):
        """
        Return true if array is compact in memory and internal size equals product
        of the shape dimensions

        有些操作会导致 strides 不再满足 row major 的形式
        1. 例如 transpose 会导致 strides 也会交换，此时就不再是 compact/contiguous
        2. slice 取出来的（slice 很灵活，可以跨着取），也不是 compact 

        Example:
          shape: (2, 3, 4)
          strides: (12, 4, 1)
          
          # transpose(1, 0)
          shape: (3, 2, 4)
          strides: (4, 12, 1)

          but compact strides is (8, 4, 1)
        """
        return self.strides == NDArray.compact_strides(self.shape) and prod(self.shape)==self._handle.size

    def compact(self):
        """ Convert a matrix to be compact Return a new"""
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device, dtype=self.dtype)
            self.device.compact(self._handle, out._handle, self.shape, self.strides, self.offset)
            return out

    def as_strided(self, shape, strides):
        """ Restride the matrix without copying memory. """
        assert len(shape) == len(strides)
        return NDArray.make(shape, strides=strides, handle=self._handle, device=self.device, dtype=self.dtype)

    def flat(self):
        return self.reshape((self.size,))

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.
        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.
        Args:
            new_shape (tuple): new shape of the array
        Returns:
            NDArray : reshaped array; this will point to the same memory as the original NDArray.
        """
        ### BEGIN YOUR SOLUTION
        if prod(new_shape) != prod(self.shape):
            raise ValueError(
                "Product of current shape is not equal to \
                              the product of the new shape!"
            )
        target = self if self.is_compact() else self.compact()
        # 可以先 compact，然后再 reshape
        # if not self.is_compact():
        #     raise ValueError("The matrix is not compact!")

        # reshape 需要重新计算 compact strides
        return NDArray.make(
            new_shape,
            strides=NDArray.compact_strides(new_shape),
            handle=target._handle,
            device=target.device,
            dtype=target.dtype,
        )
        ### END YOUR SOLUTION

    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permutation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memory as the original array.
        Args:
            new_axes (tuple): permutation order of the dimensions
        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """
        ### BEGIN YOUR SOLUTION
        assert len(new_axes) == len(self.shape), "permute must be same ndim"
        new_shape = tuple([self.shape[axes] for axes in new_axes])
        new_strides = tuple([self.strides[axes] for axes in new_axes])
        return NDArray.make(new_shape, strides=new_strides, handle=self._handle, device=self.device, dtype=self.dtype)
        ### END YOUR SOLUTION

    def swapaxes(self, axis1:int, axis2:int):
        new_shape = list(self.shape)
        new_shape[axis1], new_shape[axis2] = (new_shape[axis2], new_shape[axis1])
        new_strides = list(self.strides)
        new_strides[axis1], new_strides[axis2] = (new_strides[axis2], new_strides[axis1])
        return NDArray.make(new_shape, strides=new_strides, handle=self._handle, device=self.device, dtype=self.dtype)

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.
        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1
        Args:
            new_shape (tuple): shape to broadcast to
        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """
        ### BEGIN YOUR SOLUTION
        if self.shape == new_shape:
            return self

        new_strides = list(self.strides)
        validate_len = 0
        for x, y in zip(reversed(self.shape), reversed(new_shape)):
            assert x == y or x == 1 # 确保只有 1 不同，其余都一样
            if x != y:
                new_strides[len(new_strides) - 1 - validate_len] = 0
            validate_len += 1

        if validate_len < (n:=len(new_shape)):
            new_strides = [0] * (n - validate_len) + new_strides

        assert len(new_strides) == len(new_shape)

        return NDArray.make(
            new_shape,
            strides=tuple(new_strides),
            handle=self._handle,
            device=self.device,
            dtype=self.dtype,
        )
        ### END YOUR SOLUTION

    ### Get and set elements

    def process_slice(self, sl, dim):
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs):
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.
        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory
        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.
        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            corresponding to the subset of the matrix to get
        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memory but just
            manipulate the shape/strides/offset of the new array, referencing
            the same array as the original one.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        ### BEGIN YOUR SOLUTION
        new_shape = [(sl.stop - sl.start + sl.step - 1) // sl.step for sl in idxs] # 每个维度取多少个
        offset = sum(
            [sl.start * st for sl, st in zip(idxs, self.strides)]
        )  # 每个维度的偏移 * 每个维度的 stride 之和 --> 计算 offset
        new_strides = tuple([st * sl.step for st, sl in zip(self.strides, idxs)])  # 每个维度的 stride *= 取的步长
        return NDArray.make(
            new_shape,
            strides=tuple(new_strides),
            handle=self._handle,
            device=self.device,
            dtype=self.dtype,
            offset=offset
        )
        ### END YOUR SOLUTION

    def __setitem__(self, idxs, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view.offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view.offset,
            )

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        out = NDArray.make(self.shape, device=self.device, dtype=self.dtype)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "ewise must has same shape"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_add, self.device.scalar_add)

    __radd__ = __add__

    def __neg__(self):
        return self * (-1)

    def __mul__(self, other):
        return self.ewise_or_scalar(
                other, self.device.ewise_mul, self.device.scalar_mul
            )

    __rmul__ = __mul__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __pow__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_power, self.device.scalar_power
        )

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    ### Elementwise functions

    def log(self):
        out = NDArray.make(self.shape, device=self.device, dtype=self.dtype)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device, dtype=self.dtype)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device, dtype=self.dtype)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other):
        """Matrix multiplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.
        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will re-stride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size
        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):
            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, self.shape[1], 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device, dtype=self.dtype)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=self.device, dtype=self.dtype)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, axis):
        """Return a view to the array set up for reduction functions and output array."""
        if axis is None: # 所有轴 reduce
            view = self.reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            out = NDArray.make((1,) * self.ndim, device=self.device, dtype=self.dtype)
        else:
            if not isinstance(axis, tuple):
                axis = (axis, )
            keep_axis = tuple([a for a in range(self.ndim) if a not in axis])
            view_shape = tuple([self.shape[i] for i in keep_axis]) + (
                prod([self.shape[i] for i in axis]),
            )
            # 1. permute： 将 reduce axis 抽出来排到最后
            # 2. reshape： reduce axis 合成一个轴
            view = self.permute(keep_axis + axis).reshape(view_shape)
            out = NDArray.make(
                tuple([1 if i in axis else s for i, s in enumerate(self.shape)]),
                device=self.device,
                dtype=self.dtype
            )
        return view, out

    def sum(self, axis=None):
        view, out = self.reduce_view_out(axis)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis=None):
        view, out = self.reduce_view_out(axis)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out


def broadcast_to(array: NDArray, new_shape):
    return array.broadcast_to(new_shape)

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
    return NDArray(np.eye(n, dtype=dtype)[i], dtype=dtype, device=device)
