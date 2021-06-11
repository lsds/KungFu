import tensorflow as tf

from ._tf_oplib import _op_lib
from kungfu.python import current_rank


class Queue:
    def __init__(self, src, dst, id):
        self._src = int(src)
        self._dst = int(dst)
        self._id = int(id)

    def get(self, dtype, shape):
        return _op_lib.kungfu_queue_get(
            T=dtype,
            shape=shape,
            src=self._src,
            dst=self._dst,
            qid=self._id,
        )

    def put(self, x):
        return _op_lib.kungfu_queue_put(
            x,
            src=self._src,
            dst=self._dst,
            qid=self._id,
        )


def new_queue(src, dst):
    """new_queue creates a queue from src to dst.

        Returns None, if the current peer is not an endpoint,
        otherwise returns the queue ID.
    """
    rank = current_rank()
    if src != rank and dst != rank:
        return None
    q = _op_lib.kungfu_new_queue(src=src, dst=dst)
    return Queue(src, dst, q)
