import ctypes

class ndtype:
    def __init__(self, name, ctype):
        self.name = name
        self.ctype = ctype

    def __repr__(self):
        return f"ndtype.{self.name}"


float32 = ndtype("float32", ctypes.c_float)
float64 = ndtype("float64", ctypes.c_double)
default_dtype = float32
