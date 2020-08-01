from kungfu.python import (_get_cuda_index, current_cluster_size,
                           current_local_rank, current_local_size,
                           current_rank, run_barrier)

from . import ops, optimizers

get_cuda_index = _get_cuda_index


def nccl_built():
    return False


broadcast_parameters = ops.broadcast_parameters
