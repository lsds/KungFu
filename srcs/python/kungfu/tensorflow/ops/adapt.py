import os

from ._tf_oplib import _op_lib
from .state import counter


def get_init_checkpoint():
    """Get the initial checkpoint.

    Returns:
        A string represents the checkpoint.
    """
    # FIXME: call C API
    return os.getenv('KUNGFU_INIT_CKPT')


def resize_cluster(checkpoint, new_size):
    """Resize cluster to given size.

    Inputs:
        checkpoint: A scalar tensor of type string, new peers should be able to restore to this checkpoint.
        new_size: A scalar tensor of type int32, the new cluster size.
    Returns:
        A pair of scalar tensors (changed, keep) of type bool,
        {changed} indicates if the cluster has been changed,
        {keep} indicates if the current peer is still in the new cluster,
        the peer should quit if it is not in the new cluster.
    """
    return _op_lib.kungfu_resize_cluster(checkpoint, new_size)


def step_based_schedule(config, step=None):
    if step is None:
        step = counter()
    return _op_lib.kungfu_step_based_schedule(step,
                                              config=config,
                                              default=1,
                                              strict=False)
