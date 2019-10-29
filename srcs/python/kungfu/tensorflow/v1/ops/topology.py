from ._tf_oplib import _op_lib


def peer_info():
    """
    Returns:
        a pair of scalar tensors of int32: (rank, cluster_size).
    """
    import tensorflow as tf
    return _op_lib.kungfu_get_peer_info()
