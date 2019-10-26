import os
import sysconfig

from kungfu.config import _load_clib, _module_path

EXT_SUFFIX_KEY = 'SO'  # 'EXT_SUFFIX' does't work for python2


def _load_op_lib(name):
    suffix = sysconfig.get_config_var(EXT_SUFFIX_KEY)
    filename = os.path.join(_module_path(), name + suffix)
    import tensorflow as tf
    return tf.load_op_library(filename)


def _call_method(lib, name):
    if hasattr(lib, name):
        getattr(lib, name)()
        return True
    return False


def _load_and_init_op_lib():
    _python_lib = _load_clib('libkungfu_python')
    _call_method(_python_lib, 'kungfu_python_init')
    has_gpu = _call_method(_python_lib, 'kungfu_python_init_gpu')
    _op_lib = _load_op_lib('kungfu_tensorflow_ops')
    return _python_lib, _op_lib, has_gpu


_python_lib, _op_lib, _has_gpu = _load_and_init_op_lib()


def run_barrier():
    _python_lib.kungfu_barrier()
