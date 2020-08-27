from ._tf_oplib import _op_lib


def rank():
    """
    Returns:
        a scalar tensor of int32 representing the rank.
    """
    return _op_lib.kungfu_rank()


def cluster_size():
    """
    Returns:
        a scalar tensor of int32 representing the cluster size.
    """
    return _op_lib.kungfu_cluster_size()


def peer_info():
    """
    Returns:
        a pair of scalar tensors of int32: (rank, cluster_size).
    """
    return _op_lib.kungfu_get_peer_info()
