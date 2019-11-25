from ._tf_oplib import _op_lib


def counter(init=None):
    if init is None:
        init = 0
    return _op_lib.kungfu_counter(init=init)
