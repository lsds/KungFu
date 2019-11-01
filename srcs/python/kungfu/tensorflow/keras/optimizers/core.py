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


class KungFuKerasOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, algo, name=None):
        if name is None:
            name = "KungFu{}".format(type(optimizer).__name__)
        super(KungFuKerasOptimizer, self).__init__(name=name)
        self._optimizer = optimizer
        self._algo = algo

    def apply_gradients(self, grads_and_vars, **kwargs):
        return self._algo.apply_gradients(self._optimizer.apply_gradients, grads_and_vars, **kwargs)

    def get_config(self):
        return self._optimizer.optimizer.get_config()
