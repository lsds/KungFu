import torch


def import_op_lib():
    import kungfu_torch_ops as ops
    return ops


ops = import_op_lib()

all_reduce_op_map = {
    'torch.FloatTensor': ops.all_reduce_cpu,
}

if hasattr(ops, 'all_reduce_cuda'):
    all_reduce_op_map['torch.cuda.FloatTensor'] = ops.all_reduce_cuda
