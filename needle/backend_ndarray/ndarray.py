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
            self.device = array.device if device is None else device
            self.dtype = array.dtype if dtype is None else dtype
        elif isinstance(array, np.ndarray):
            NDArray.make(
                array.shape,
                self = self,
                device = device,
                dtype = NDArray.map_dtype(array.dtype) if dtype is None else dtype,
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
        if dtype == np.float32:
            return float32
        elif dtype == np.float64:
            return float64
        else:
            raise Exception(f"only support np.float32 and np.float64, but input dtype is {dtype}")

    # use numpy.array replace this method
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
        def flatten_recursively(nested_list):
            flat_data = []
            shape = []
            if isinstance(nested_list, list):
                for sublist in nested_list:
                    inner_data, inner_shape = flatten_recursively(sublist)
                    flat_data.extend(inner_data)
                shape.append(len(nested_list))
                shape.extend(inner_shape)
            else:
                flat_data.append(nested_list)
            return flat_data, shape

        flat_data, shape = flatten_recursively(nested_list)
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
        array.dtype = default_dtype if dtype is None else dtype
        array.offset = offset
        # cpp 中开辟一块对齐的内存地址
        array._handle = array.device.Array(array.size) if handle is None else handle
        return array

    @property
    def size(self):
        return prod(self.shape)

    def numpy(self):
        """convert to a numpy array"""
        return self.device.to_numpy(
            self._handle, list(self.shape), list(self.strides), self.offset
        )

    def __repr__(self):
        return f"NDArray({str(self.numpy())}, dtype={self.dtype}, device={self.device})"

    def __str__(self):
        return str(self.numpy())

    def reshape(self):
        pass

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        out = NDArray.make(self.shape, device=self.device, dtype=self.dtype)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "ewise must has same shape"
            ewise_func(self._handle, other._handle ,out._handle)
        else:
            scalar_func(self._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_add, self.device.scalar_add)

    __radd__ = __add__
