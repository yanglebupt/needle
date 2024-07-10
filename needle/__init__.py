import sys
sys.path.append("./backend_ndarray")

from . import ops
from .ops import *
from .autograd import Tensor

from .init import ones, zeros, zeros_like, ones_like

from . import init
from . import data
from . import nn
from . import optim
from . import backend_ndarray

from .backend_ndarray import *
