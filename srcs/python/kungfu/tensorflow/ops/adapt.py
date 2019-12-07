import os

from ._tf_oplib import _op_lib
from .state import counter


def _get_init_cluster_version_id():
    """Get the initial cluster version id.

    Returns:
        A string represents the cluster version.
    """
    # FIXME: call C API
    return os.getenv('KUNGFU_INIT_CKPT')


def _resize_cluster(cluster_version_id, new_size, debug=False):
    return _op_lib.kungfu_resize_cluster(cluster_version_id,
                                         new_size,
                                         debug=debug)


def resize_cluster(new_size, debug=False):
    """Resize cluster to given size.

    Inputs:
        new_size: A scalar tensor of type int32, the new cluster size.
    Returns:
        A pair of scalar tensors (changed, keep) of type bool,
        {changed} indicates if the cluster has been changed,
        {keep} indicates if the current peer is still in the new cluster,
        the peer should quit if it is not in the new cluster.
    """
    # Declare a cluster version id counter.
    init_cluster_version_id = int(_get_init_cluster_version_id())
    cluster_version_counter = counter(init_cluster_version_id)

    # The cluster version id is increased by 1 everytime you call resize
    next_cluster_version_id = tf.as_string(cluster_version_counter + 1)

    return _resize_cluster(next_cluster_version_id, new_size, debug)


def step_based_schedule(config, step=None):
    if step is None:
        step = counter()
    return _op_lib.kungfu_step_based_schedule(step,
                                              config=config,
                                              default=1,
                                              strict=False)
