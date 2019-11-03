import os
import sysconfig

from kungfu.loader import _module_path

EXT_SUFFIX_KEY = 'SO'  # 'EXT_SUFFIX' does't work for python2


def _load_op_lib(name):
    suffix = sysconfig.get_config_var(EXT_SUFFIX_KEY)
    filename = os.path.join(_module_path(), name + suffix)
    import tensorflow as tf
    return tf.load_op_library(filename)


def _load_and_init_op_lib():
    _op_lib = _load_op_lib('kungfu_tensorflow_ops')
    return _op_lib


_op_lib = _load_and_init_op_lib()
