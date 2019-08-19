import tensorflow as tf


class BaseDatasetAdaptor(object):
    def __init__(self, batch_size=10, shard_count=1, shard_id=0):
        self._batch_size = tf.Variable(tf.constant(batch_size, tf.int64))
        self._shard_count = tf.Variable(tf.constant(shard_count, tf.int64))
        self._shard_id = tf.Variable(tf.constant(shard_id, tf.int64))
        self._offset = tf.Variable(tf.constant(0, tf.int64))

    def create_update_offset(self):
        return tf.assign_add(self._offset,
                             self._batch_size * self._shard_count)

    def create_rewind(self):
        return tf.assign(self._offset, 0)

    def create_update_topology(self):
        # Subclass should implement
        raise RuntimeError('Not implemented')

    def debug(self, sess):
        off, np, rank, bs = sess.run([
            self._offset, self._shard_count, self._shard_id, self._batch_size
        ])
        print('offset=%d, np=%d, rank=%d, batch size=%d' % (off, np, rank, bs))

    def __call__(self, ds):
        ds = ds.skip(self._offset)
        ds = ds.batch(self._batch_size)
        ds = ds.shard(self._shard_count, self._shard_id)
        it = ds.make_initializable_iterator()
        return it.initializer, it.get_next()


class ExampleDatasetAdaptor(BaseDatasetAdaptor):
    def create_update_topology(self):
        update_cluster = tf.assign_add(self._shard_count, 1)
        update_batch_size = tf.assign(self._batch_size, self._batch_size + 2)
        with tf.control_dependencies([update_cluster]):
            return tf.group([
                tf.assign(self._shard_id,
                          tf.mod(self._shard_id + 1, self._shard_count)),
                update_batch_size,
            ])
