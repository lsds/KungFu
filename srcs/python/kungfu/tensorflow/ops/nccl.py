import tensorflow as tf
from kungfu._utils import map_maybe

from ._tf_oplib import _op_lib
from .topology import peer_info


def _begin_nccl_ops(*args, **kwargs):
    if hasattr(_op_lib, 'kungfu_begin_nccl_ops'):
        return _op_lib.kungfu_begin_nccl_ops(*args, **kwargs)
    else:
        raise RuntimeError(
            "KungFu is not installed with NCCL. Please reinstall with KUNGFU_ENABLE_NCCL=1"
        )


def _scheduled_nccl_all_reduce_v2(t, op_name=None):
    if op_name is None:
        op_name = t.name
    return _op_lib.kungfu_scheduled_nccl_all_reduce_v2(t, op_name=op_name)


def group_nccl_all_reduce_v2(ts):
    names = [t.name for t in ts if t is not None]
    names = list(sorted(names))
    with tf.control_dependencies([
            _begin_nccl_ops(names, scope='global'),
    ]):
        return map_maybe(_scheduled_nccl_all_reduce_v2, ts)
