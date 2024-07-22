import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, shape=None, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt( 6 / (fan_in + fan_out) )
    fin_shape = (fan_in, fan_out) if shape is None else shape
    return rand(*fin_shape, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, shape=None, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt( 2 / (fan_in + fan_out) )
    fin_shape = (fan_in, fan_out) if shape is None else shape
    return randn(*fin_shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", shape=None, **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    a = math.sqrt( 2 * 3 / fan_in )
    fin_shape = (fan_in, fan_out) if shape is None else shape
    return rand(*fin_shape, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", shape=None, **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = math.sqrt( 2 / fan_in )
    fin_shape = (fan_in, fan_out) if shape is None else shape
    return randn(*fin_shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION
