import torch


def import_op_lib():
    import kungfu_torch_ops as ops
    return ops


ops = import_op_lib()

all_reduce_op_map = {
    'torch.FloatTensor': ops.all_reduce_cpu,
}

all_gather_op_map = {
    'torch.FloatTensor': ops.all_gather_cpu,
}

all_reduce_async_op_map = {}
broadcast_async_op_map = {}

if hasattr(ops, 'all_reduce_cuda'):
    all_reduce_op_map['torch.cuda.FloatTensor'] = ops.all_reduce_cuda

if hasattr(ops, 'all_reduce_cuda_async'):
    all_reduce_async_op_map[
        'torch.cuda.FloatTensor'] = ops.all_reduce_cuda_async

if hasattr(ops, 'broadcast_cuda_async'):
    broadcast_async_op_map[
        'torch.cuda.FloatTensor'] = ops.broadcast_cuda_async

if hasattr(ops, 'all_gather_cuda'):
    all_gather_op_map[
        'torch.cuda.FloatTensor'] = ops.all_gather_cuda
