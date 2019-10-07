from .loader import _op_lib


def barrier():
    return _op_lib.kungfu_barrier()


def broadcast(t):
    return _op_lib.broadcast(t)


def all_reduce(t):
    return _op_lib.all_reduce(t, input_tensor_name=t.name)


def all_reduce_gpu(t):
    return _op_lib.all_reduce_gpu(t, input_tensor_name=t.name)


def global_variance(t):
    return _op_lib.global_variance(t)


def start_gpu_group(*args, **kwargs):
    return _op_lib.start_gpu_group(*args, **kwargs)


def cpu_group_all_reduce(ts):
    return [all_reduce(t) for t in ts]


def gpu_group_all_reduce(ts):
    names = [t.name for t in ts]
    names = list(sorted(names))  # FIXME: use topsort
    import tensorflow as tf
    with tf.control_dependencies([
            start_gpu_group(names),
    ]):
        return [all_reduce_gpu(t) for t in ts]


def group_all_reduce(ts, nccl=False):
    # FIXME: auto determine device
    if nccl:
        print('Try to use GPU NCCL to perform all-reduce')
        return gpu_group_all_reduce(ts)
    print('Try to use KungFu MPI to perform all-reduce')
    return cpu_group_all_reduce(ts)
