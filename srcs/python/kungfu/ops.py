import os
import sysconfig


def _load_op_lib(name):
    module_path = os.path.dirname(__file__)
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    filename = os.path.join(module_path, name + suffix)
    import tensorflow as tf
    return tf.load_op_library(filename)


_op_lib_name = 'kungfu_tensorflow_ops'
_op_lib = None


def _lazy_load_op_lib():
    global _op_lib
    if _op_lib is None:
        _op_lib = _load_op_lib(_op_lib_name)
    return _op_lib


def init_kungfu():
    _op_lib = _lazy_load_op_lib()
    return _op_lib.init_kungfu()


def broadcast(t):
    _op_lib = _lazy_load_op_lib()
    return _op_lib.broadcast(t)


def all_reduce(t):
    _op_lib = _lazy_load_op_lib()
    return _op_lib.all_reduce(t)


def global_variance(t):
    _op_lib = _lazy_load_op_lib()
    return _op_lib.global_variance(t)


def global_step_modifier(step):
    _op_lib = _lazy_load_op_lib()
    return _op_lib.global_step_modifier(step)


def set_num_gradients(n):
    _op_lib = _lazy_load_op_lib()
    return _op_lib.set_num_gradients(n)
