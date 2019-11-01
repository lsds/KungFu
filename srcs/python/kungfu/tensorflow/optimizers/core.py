import tensorflow as tf
from kungfu.tensorflow import _tf_optimizer
from kungfu.tensorflow.v1.ops import counter


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


class KungFuOptimizer(_tf_optimizer, tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, name=None, use_locking=False):
        if name is None:
            name = "KungFu{}".format(type(optimizer).__name__)

        if isinstance(optimizer, _tf_optimizer):
            super(KungFuOptimizer, self).__init__(name=name,
                                                  use_locking=use_locking)
        elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
            super(KungFuOptimizer, self).__init__(name=name)
        else:
            raise RuntimeError('Cannot wrap: %s' % type(optimizer).__name__)

        self._optimizer = optimizer

    # tf.train.Optimizer, tf.keras.optimizers.Optimizer
    def apply_gradients(self, *args, **kwargs):
        raise RuntimeError('apply_gradients should be called in a subclass.')

    # tf.train.Optimizer
    def compute_gradients(self, *args, **kwargs):
        return self._optimizer.compute_gradients(*args, **kwargs)

    # tf.train.Optimizer
    def get_slot(self, *args, **kwargs):
        return self._optimizer.get_slot(*args, **kwargs)

    # tf.train.Optimizer
    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    # tf.train.Optimizer
    def variables(self, *args, **kwargs):
        return self._optimizer.variables(*args, **kwargs)

    # tf.keras.optimizers.Optimizer
    def get_config(self):
        return self._optimizer.optimizer.get_config()
