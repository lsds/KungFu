from ._tf_oplib import _op_lib


def fake_error(x):
    return _op_lib.kungfu_fake_error(x)
