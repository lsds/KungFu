import os
import platform
from ctypes import cdll


def _module_path():
    dirname = os.path.dirname
    return dirname(__file__)


def _load_clib(name):
    suffix = 'so' if platform.uname()[0] != 'Darwin' else 'dylib'
    filename = os.path.join(_module_path(), name + '.' + suffix)
    return cdll.LoadLibrary(filename)


def _call_method(lib, name, force=False):
    if hasattr(lib, name):
        getattr(lib, name)()
        return True
    if force:
        raise RuntimeError('failed to call %s' % name)
    return False


def _call_method_with(lib, name, *args):
    if hasattr(lib, name):
        getattr(lib, name)(*args)
    else:
        raise RuntimeError('failed to call %s' % name)
