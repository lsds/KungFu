from .loader import _init_lib, _op_lib


def current_rank():
    return _init_lib.kungfu_rank()


def current_cluster_size():
    return _init_lib.kungfu_cluster_size()


def _get_other_ranks():
    self_rank = current_rank()
    ranks = list(range(current_cluster_size()))
    return [r for r in ranks if r != self_rank]


def peer_info():
    """
    Returns:
        a pair of scalar tensors of int32: (rank, cluster_size).
    """
    import tensorflow as tf
    version = tf.constant(-1, dtype=tf.int32)
    return _op_lib.kungfu_get_peer_info(version)
