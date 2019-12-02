from ._tf_oplib import _op_lib


def fpga_all_reduce(t):
    """Create a new all_reduce operator for given tensor."""
    return _op_lib.kungfu_fpga_all_reduce(t)
