import tensorflow as tf

from kungfu.datasets.adaptor import BaseDatasetAdaptor
from kungfu.ops import peer_info


class DynamicDatasetAdaptor(BaseDatasetAdaptor):
    def __init__(self, batch_size=10):
        rank, np = peer_info(tf.constant(-1, dtype=tf.int32))
        self._rank = rank
        self._np = np
        self._batch_size = tf.Variable(tf.constant(batch_size, tf.int64),
                                       trainable=False)
        self._shard_count = tf.Variable(tf.cast(np, tf.int64), trainable=False)
        self._shard_id = tf.Variable(tf.cast(rank, tf.int64), trainable=False)
        self._offset = tf.Variable(tf.constant(0, tf.int64))
