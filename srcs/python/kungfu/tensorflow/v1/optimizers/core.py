import tensorflow as tf


def fuse(ts):
    return tf.concat([tf.reshape(t, [-1]) for t in ts], -1)


def defuse(y, shapes):
    ts = []
    off = 0
    for s in shapes:
        size = s.num_elements()
        x = tf.slice(y, [off], [size])
        x = tf.reshape(x, s)
        ts.append(x)
        off += size
    if off != y.shape.num_elements():
        raise RuntimeError('invalid shapes')
    return ts


class KungFuOptimizer(tf.train.Optimizer):
    """An optimizer that would negotiate the gradients before apply it."""
    def __init__(self, optimizer, name=None, use_locking=False):
        if name is None:
            name = "KungFu{}".format(type(optimizer).__name__)
        super(KungFuOptimizer, self).__init__(name=name,
                                              use_locking=use_locking)
        self._optimizer = optimizer
        self._step = tf.Variable(0, trainable=False, dtype=tf.int32)

    def _distributed_initializer(self):
        raise RuntimeError('_distributed_initializer is not implemented.')

    def compute_gradients(self, *args, **kwargs):
        self._init_op = tf.cond(tf.equal(self._step, 0),
                                self._distributed_initializer, tf.no_op)
        with tf.control_dependencies([self._init_op]):
            with tf.control_dependencies([tf.assign_add(self._step, 1)]):
                return self._optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.apply_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.variables(*args, **kwargs)
