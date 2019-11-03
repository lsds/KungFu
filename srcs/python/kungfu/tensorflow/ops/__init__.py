from kungfu.ext import (_get_other_ranks, _has_gpu, current_cluster_size,
                        current_rank, run_barrier)

from ._tf_oplib import _op_lib
from .adapt import get_init_checkpoint, resize_cluster
from .collective import (all_reduce, barrier, broadcast, group_all_reduce,
                         group_nccl_all_reduce)
from .local import save_variable, save_variables
from .monitor import global_noise_scale
from .p2p import request_variable, request_variable_with_template
from .state import counter
from .topology import peer_info

__all__ = [
    'barrier',
    'broadcast',
    'group_all_reduce',
]


def fuse(ts):
    import tensorflow as tf
    return tf.concat([tf.reshape(t, [-1]) for t in ts], -1)


def defuse(y, shapes):
    import tensorflow as tf
    ts = []
    off = 0
    for s in shapes:
        size = s.num_elements()
        x = tf.slice(y, [off], [size])
        x = tf.reshape(x, s)
        ts.append(x)
        off += size
    if off != y.shape.num_elements():
        raise RuntimeError('invalid shapes')
    return ts


def get_peer_latencies(local_step=None):
    """Returns the vector V of round-trip time from this peer to all other peers.

    For the peer of rank i, V[j] is the RTT from i to j (j != i), V[i] = 0.
    """
    # FIXME: don't require input
    if local_step is None:
        import tensorflow as tf
        local_step = tf.Variable(tf.zeros([], tf.int64), trainable=False)
    return _op_lib.kungfu_get_peer_latencies(
        local_step, cluster_size=current_cluster_size())


def global_minimum_spanning_tree(self_weights):
    """Compute the minimum spanning tree.

    self_weights: a vector of length n,
        where n is the number of peers in the cluster.
        All self_weights vectors from n peers are gathered to a matrix W of
        n x n. The MST is then computed based on (W + W^T)/2.
    returns:
        a matrix m of (n - 1) x 2,
        where (m[i][0], m[i][1]) is the i-th edge of the tree.
    """
    return _op_lib.kungfu_minimum_spanning_tree(self_weights)


def get_neighbour_mask(edges):
    """Compute a bool vector of neighbours for the current peer.

    For the peer of rank i, v[j] = true if (i, j) is an edge of the MST,
    otherwise v[j] = false.
    """
    return _op_lib.kungfu_get_neighbour_mask(
        edges, self_rank=current_rank(), cluster_size=current_cluster_size())


def round_robin(mask):
    return _op_lib.kungfu_round_robin(mask)
