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
            name = "KungFuOptimizer{}".format(type(optimizer).__name__)
        super(KungFuOptimizer, self).__init__(name=name,
                                              use_locking=use_locking)
        self._optimizer = optimizer
        self._model_variables = None

    # distributed_initializer must be called after minimize
    def distributed_initializer(self):
        raise RuntimeError(
            'Distributed optimizers must implement how variables are replicated.'
        )

    def model_variables(self):
        if not self._model_variables:
            raise RuntimeError('minimize or compute_gradients is NOT called')
        return self._model_variables

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients and negotiate with peers."""
        grads_and_vars = self._optimizer.compute_gradients(*args, **kwargs)

        # An optimizer could minimize variables other than tf.trainable_variables
        # It is safer to get the correct list of variables that need synchornisation here
        if self._model_variables is None:
            self._model_variables = [v for g, v in grads_and_vars]

        return grads_and_vars

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
