import tensorflow as tf
from kungfu.tensorflow.v1.ops import counter
from kungfu.tensorflow import _tf_optimizer


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


class KungFuOptimizer(_tf_optimizer):
    def __init__(self, optimizer, name=None, use_locking=False):
        if name is None:
            name = "KungFu{}".format(type(optimizer).__name__)
        super(KungFuOptimizer, self).__init__(name=name,
                                              use_locking=use_locking)
        self._optimizer = optimizer

    def compute_gradients(self, *args, **kwargs):
        return self._optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, *args, **kwargs):
        raise RuntimeError('apply_gradients should be called in a subclass.')

    def get_slot(self, *args, **kwargs):
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self._optimizer.variables(*args, **kwargs)
