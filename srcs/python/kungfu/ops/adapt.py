import os

from .loader import _init_lib, _op_lib


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
        checkpoint: string, new peers should be able to restore to this checkpoint.
        new_size: int, the new cluster size.
    Returns:
        A bool indicates if the current peer should quit.
    """
    return _op_lib.kungfu_resize_cluster(checkpoint, new_size)


# The following APIs are deprecated.


def start_step():  # temporary API for experiment
    return _init_lib.kungfu_start_step()


def get_init_version():
    """Returns a non-negative integer representing the cluster version."""
    init_sess = os.getenv('KUNGFU_INIT_SESS')
    version = int(init_sess)
    if version < 0:
        raise RuntimeError('invalid version')
    return version


def get_start_step(version):
    """
    Input:
        version: A scalar tensor of int32,
    Returns:
        a scalar tensors of int64, the start global step
    """
    return _op_lib.kungfu_get_start_step(version)


def propose_update(target_global_step, target_version, new_size):
    """
    Inputs:
        target_global_step: a scalar tensor of int64
        target_version: a scalar tensor of int32
        new_size: a scalar tensor of int32
    Returns:
        a pair of scalar tensors of bool: (accepted, keep)
        accepted: indicates if proposal is accepts
        keep: indicates if self is still in the new cluster
    """
    return _op_lib.kungfu_propose_update(target_global_step, target_version,
                                         new_size)


def update_cluster(version):
    """Returns a bool scalar which indicates if this peer is still in the cluster."""
    return _op_lib.kungfu_update_cluster(version)
