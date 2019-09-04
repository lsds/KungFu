import tensorflow as tf


class KungFuOptimizer(tf.train.Optimizer):
    """An optimizer that would negotiate the gradients before apply it."""
    def __init__(self, optimizer, name=None, use_locking=False):
        if name is None:
            name = "KungFuOptimizer{}".format(type(optimizer).__name__)
        super(KungFuOptimizer, self).__init__(name=name,
                                              use_locking=use_locking)

        self._optimizer = optimizer

    # get_initializer must be called after minimize
    def get_initializer_op(self):
        if not self._grads_and_vars:
            raise RuntimeError('compute_gradients NOT called')
        return self._get_initializer_op(self._grads_and_vars)

    # The subclass should implement this with its own negotiation strategy
    def _negotiate_grads_by_strategy(self, grads_and_vars):
        raise RuntimeError('Not implemented')

    # The subclass can override this
    def _get_initializer_op(self, grads_and_vars):
        return tf.no_op()

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients and negotiate with peers."""
        grads_and_vars = self._optimizer.compute_gradients(*args, **kwargs)
        self._grads_and_vars = grads_and_vars
        grads_and_vars_to_negotiate = [(g, v) for g, v in grads_and_vars
                                       if g is not None]
        return self._negotiate_grads_by_strategy(grads_and_vars_to_negotiate)

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
