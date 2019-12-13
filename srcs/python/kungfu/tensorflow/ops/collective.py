import tensorflow as tf
from kungfu._utils import map_maybe

from ._tf_oplib import _op_lib
from .topology import peer_info


def barrier():
    """Create a new barrier operator."""
    return _op_lib.kungfu_barrier()


def consensus(t):
    return _op_lib.kungfu_consensus(t, tensor_name=t.name, strong=True)


def broadcast(t):
    """Create a new broadcast operator for given tensor."""
    return _op_lib.kungfu_broadcast(t)


def all_reduce(t):
    """Create a new all_reduce operator for given tensor."""
    return _op_lib.kungfu_all_reduce(t)


def _maybe_group_all_reduce(ts, group_all_reduce_fn):
    # a helper function to bypass all_reduce for np = 1
    _rank, np = peer_info()
    import tensorflow as tf
    return tf.cond(np > 1, lambda: group_all_reduce_fn(ts),
                   lambda: [tf.identity(t) for t in ts])


def group_all_reduce(ts):
    """Create a list of all_reduce operators for given tensor list."""
    return map_maybe(all_reduce, ts)


def _nccl_all_reduce(t):
    return _op_lib.kungfu_nccl_all_reduce(t, input_tensor_name=t.name)


def _scheduled_nccl_all_reduce(t):
    return _op_lib.kungfu_scheduled_nccl_all_reduce(t,
                                                    input_tensor_name=t.name)


def _start_nccl_scheduler(*args, **kwargs):
    if hasattr(_op_lib, 'kungfu_start_nccl_scheduler'):
        return _op_lib.kungfu_start_nccl_scheduler(*args, **kwargs)
    else:
        raise RuntimeError("KungFu is not installed with NCCL.")


def group_nccl_all_reduce(ts):
    """Create a list of all_reduce operators for given tensor list, using NCCL."""
    names = [t.name for t in ts if t is not None]
    if len(names) == 1:
        return map_maybe(_nccl_all_reduce, ts)  # exactly one of ts is not None
    else:
        print("WARNING: Please fuse %d tensors before using NCCL." %
              len(names))
        names = list(sorted(names))  # FIXME: use topsort
        import tensorflow as tf
        with tf.control_dependencies([
                _start_nccl_scheduler(names),
        ]):
            return map_maybe(_scheduled_nccl_all_reduce, ts)
