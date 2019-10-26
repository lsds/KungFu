import os

from .loader import _op_lib, _python_lib


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
        A scalar tensor of type bool indicates if the current peer is still in the new cluster,
        the peer should quit if false.
    """
    return _op_lib.kungfu_resize_cluster(checkpoint, new_size)
