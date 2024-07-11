import ndarray_backend_cpu

cuda_instance = None
cpu_instance = None

class BackendDevice:
    def __init__(self, name, module):
        """use backend device to fetch array operation api
        Args:
            name (string): name of device
            module (module): cpp pybind11 module for array operation in specific device
        Returns:
            None
        """
        self.name = name
        self.module = module

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def enabled(self):
        return self.module is not None

    def __repr__(self) -> str:
        return self.name + "()"

    def __getattr__(self, name):
        """key method for fetch array operation api"""
        return getattr(self.module, name)

def cuda():
    global cuda_instance
    if cuda_instance is None:
        cuda_instance = BackendDevice("cuda", None)
    return cuda_instance

def cpu():
    global cpu_instance
    if cpu_instance is None:
        cpu_instance = BackendDevice("cpu", ndarray_backend_cpu)
    return cpu_instance

def default_device():
    return cpu()


def all_devices():
    return [cpu(), cuda()]
