from ._tf_oplib import _op_lib


def counter(init=None, incr=None, debug=False):
    if init is None:
        init = 0
    if incr is None:
        incr = 1
    return _op_lib.kungfu_counter(init=init, incr=incr, debug=debug)


def exponential_moving_average(x, alpha=0.9):
    return _op_lib.kungfu_exponential_moving_average(x, alpha=alpha)
