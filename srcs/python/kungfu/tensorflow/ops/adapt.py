from ._tf_oplib import _op_lib
from .state import counter


def resize_cluster_from_url():
    """Resize cluster from config server.

    Returns:
        A pair of scalar tensors (changed, keep) of type bool,
        {changed} indicates if the cluster has been changed,
        {keep} indicates if the current peer is still in the new cluster,
        the peer should quit if it is not in the new cluster.
    """

    resize_op = _op_lib.kungfu_resize_cluster_from_url()
    if hasattr(_op_lib, 'kungfu_reset_nccl_helper'):
        changed, keep = resize_op
        return _op_lib.kungfu_reset_nccl_helper(changed, keep)
    else:
        return resize_op


def step_based_schedule(config, step=None):
    if step is None:
        step = counter()
    return _op_lib.kungfu_step_based_schedule(step,
                                              config=config,
                                              default=1,
                                              strict=False)


def set_tree(tree):
    """Set the default communication tree.

    Inputs:
        tree: an int32 tensor with shape [n], where
            - n is the number of peers in the current cluster;
            - tree[i] is the father of i if tree[i] != i;
            - i is the root if tree[i] == i.
    """
    return _op_lib.kungfu_set_tree(tree)
