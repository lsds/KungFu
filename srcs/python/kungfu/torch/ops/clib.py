import torch


def import_op_lib():
    import kungfu_torch_ops as ops
    return ops


ops = import_op_lib()
