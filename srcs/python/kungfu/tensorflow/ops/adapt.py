from ._tf_oplib import _op_lib
from .state import counter


def resize_cluster_from_url():
    """Resize cluster from config server.

    Returns:
        A pair of scalar tensors (changed, detached) of type bool,
        {changed} indicates if the cluster has been changed,
        {detached} indicates if the current peer is detached from the old cluster,
        the peer should quit if it is not in the new cluster.
    """

    resize_op = _op_lib.kungfu_resize_cluster_from_url()
    if hasattr(_op_lib, 'kungfu_reset_nccl_helper'):
        changed, detached = resize_op
        return _op_lib.kungfu_reset_nccl_helper(changed, detached)
    else:
        return resize_op


def step_based_schedule(config, step=None):
    if step is None:
        step = counter()
    return _op_lib.kungfu_step_based_schedule(step,
                                              config=config,
                                              default=1,
                                              strict=False)


def resize(n):
    """Resize the cluster to n.

    Inputs:
        n: A scalar tensor of uint32.
    Returns:
        A scalar tensor of bool, indicates if the cluster has been changed.
    """
    changed, detached = _op_lib.kungfu_resize_cluster(n)
    if hasattr(_op_lib, 'kungfu_reset_nccl_helper'):
        changed, detached = _op_lib.kungfu_reset_nccl_helper(changed, detached)
        return changed
    else:
        return changed


def set_tree(tree):
    """Set the default communication tree.

    Inputs:
        tree: an int32 tensor with shape [n], where
            - n is the number of peers in the current cluster;
            - tree[i] is the father of i if tree[i] != i;
            - i is the root if tree[i] == i.
    """
    return _op_lib.kungfu_set_tree(tree)


def calc_stats():
    """Calculate key communication stratetgy metrics based on current state."""
    return _op_lib.kungfu_calc_stats()
