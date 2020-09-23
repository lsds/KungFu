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


def all_reduce(t, op='sum'):
    """Create a new all_reduce operator for given tensor."""
    return _op_lib.kungfu_all_reduce(t, op=op)


def monitored_all_reduce(t, tree=None, op='sum'):
    """Create a new all_reduce operator for given tensor and topology.

    Inputs:
        t: the tensor for allreduce
        tree: an int32 tensor with shape [n], where
            - n is the number of peers in the current cluster;
            - tree[i] is the father of i if tree[i] != i;
            - i is the root if tree[i] == i.
    """
    if tree is None:
        tree = []
    else:
        print(
            'calling monitored_all_reduce with tree is deprecated, please use set_tree API'
        )
    # TODO: return monitoring metrics
    return _op_lib.kungfu_monitored_all_reduce(t, tree, op=op)


def all_gather(t):
    """Create a new all_gather operator for given tensor.

    Inputs:
        A tensor of any shape. The shape must be consistent on all peers.

    Returns:
        A tensor with leading dimension equal to the number of peers,
        and the rest dimensions equal to the dimensions in the original shape.
    """
    return _op_lib.kungfu_all_gather(t)


def _maybe_group_all_reduce(ts, group_all_reduce_fn):
    # a helper function to bypass all_reduce for np = 1
    _rank, np = peer_info()
    return tf.cond(np > 1, lambda: group_all_reduce_fn(ts),
                   lambda: [tf.identity(t) for t in ts])


def group_all_reduce(ts):
    """Create a list of all_reduce operators for given tensor list."""
    return map_maybe(all_reduce, ts)


def _nccl_all_reduce(t):
    return _op_lib.kungfu_nccl_all_reduce(t)


def _scheduled_nccl_all_reduce(t, op_name=None):
    if op_name is None:
        op_name = t.name
    return _op_lib.kungfu_scheduled_nccl_all_reduce(t, op_name=op_name)


def _scheduled_hierarchical_nccl_all_reduce(t, op_names):
    return _op_lib.kungfu_scheduled_hierarchical_nccl_all_reduce(
        t, op_names=op_names)


def _start_nccl_scheduler(*args, **kwargs):
    if hasattr(_op_lib, 'kungfu_start_nccl_scheduler'):
        return _op_lib.kungfu_start_nccl_scheduler(*args, **kwargs)
    else:
        raise RuntimeError(
            "KungFu is not installed with NCCL. Please reinstall with KUNGFU_ENABLE_NCCL=1"
        )


def group_nccl_all_reduce(ts):
    """Create a list of all_reduce operators for given tensor list, using NCCL."""
    names = [t.name for t in ts if t is not None]
    if len(names) == 1:
        return map_maybe(_nccl_all_reduce, ts)  # exactly one of ts is not None
    else:
        names = list(sorted(names))
        with tf.control_dependencies([
                _start_nccl_scheduler(names, scope='global'),
        ]):
            return map_maybe(_scheduled_nccl_all_reduce, ts)


def group_hierarchical_nccl_all_reduce(ts):
    names = [t.name for t in ts if t is not None]

    def reduce_op_name(name):
        return 'reduce_' + name

    def bcast_op_name(name):
        return 'bcast_' + name

    reduce_names = map_maybe(lambda t: reduce_op_name(t.name), ts)
    bcast_names = map_maybe(lambda t: bcast_op_name(t.name), ts)

    def all_reduce(args):
        i, t = args
        return _scheduled_hierarchical_nccl_all_reduce(
            t, op_names=[reduce_names[i], bcast_names[i]])

    t_names = list(sorted(names))
    all_op_names = list([reduce_op_name(name) for name in t_names] +
                        [bcast_op_name(name) for name in t_names])

    with tf.control_dependencies([
            _start_nccl_scheduler(all_op_names, scope='local'),
    ]):
        return map_maybe(all_reduce, enumerate(ts))
