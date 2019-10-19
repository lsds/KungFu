from .loader import _op_lib
from .topology import peer_info


def barrier():
    return _op_lib.kungfu_barrier()


def broadcast(t):
    return _op_lib.broadcast(t)


def all_reduce(t):
    return _op_lib.all_reduce(t, input_tensor_name=t.name)


def _group_all_reduce(ts):
    return [all_reduce(t) for t in ts]


def _maybe_group_all_reduce(ts, group_all_reduce_fn):
    # a helper function to bypass all_reduce for np = 1
    _rank, np = peer_info()
    import tensorflow as tf
    return tf.cond(np > 1, lambda: group_all_reduce_fn(ts),
                   lambda: [tf.identity(t) for t in ts])


def group_all_reduce(ts):
    return _maybe_group_all_reduce(ts, _group_all_reduce)


def _nccl_all_reduce(t):
    return _op_lib.all_reduce_gpu(t, input_tensor_name=t.name)


def _start_nccl_scheduler(*args, **kwargs):
    return _op_lib.start_gpu_group(*args, **kwargs)


def _group_nccl_all_reduce(ts):
    names = [t.name for t in ts]
    names = list(sorted(names))  # FIXME: use topsort
    import tensorflow as tf
    with tf.control_dependencies([
            _start_nccl_scheduler(names),
    ]):
        return [_nccl_all_reduce(t) for t in ts]


def group_nccl_all_reduce(ts):
    return _maybe_group_all_reduce(ts, _group_nccl_all_reduce)
