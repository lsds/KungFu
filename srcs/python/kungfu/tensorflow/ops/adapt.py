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
    return _op_lib.kungfu_resize_cluster_from_url()


def step_based_schedule(config, step=None):
    if step is None:
        step = counter()
    return _op_lib.kungfu_step_based_schedule(step,
                                              config=config,
                                              default=1,
                                              strict=False)


def set_tree(tree):
    return _op_lib.kungfu_set_tree(tree)
