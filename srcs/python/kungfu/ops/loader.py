import os
import platform
import sysconfig
from ctypes import cdll

EXT_SUFFIX_KEY = 'SO'  # 'EXT_SUFFIX' does't work for python2


def _load_op_lib(name):
    module_path = os.path.dirname(__file__)
    suffix = sysconfig.get_config_var(EXT_SUFFIX_KEY)
    filename = os.path.join(module_path, name + suffix)
    import tensorflow as tf
    return tf.load_op_library(filename)


def _load_init_lib(name):
    module_path = os.path.dirname(__file__)
    suffix = 'so' if platform.uname()[0] != 'Darwin' else 'dylib'
    filename = os.path.join(module_path, name + '.' + suffix)
    return cdll.LoadLibrary(filename)


def _call_method(lib, name):
    if hasattr(lib, name):
        getattr(lib, name)()
        return True
    return False


def _load_and_init_op_lib():
    _op_lib = _load_op_lib('kungfu_tensorflow_ops')
    _init_lib = _load_init_lib('libkungfu_tensorflow_init')
    _call_method(_init_lib, 'kungfu_tensorflow_init')
    has_gpu = _call_method(_init_lib, 'kungfu_tensorflow_init_gpu')
    return _op_lib, _init_lib, has_gpu


_op_lib, _init_lib, _has_gpu = _load_and_init_op_lib()
