import tensorflow as tf


class KungFuOptimizer(tf.train.Optimizer):
    """An optimizer that would negotiate the gradients before apply it."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse=''):
        if name is None:
            name = "KungFuOptimizer{}".format(type(optimizer).__name__)
        super(KungFuOptimizer, self).__init__(
            name=name, use_locking=use_locking)

        self._optimizer = optimizer
        self._device_dense = device_dense
        self._device_sparse = device_sparse

        self._enable_set_num_gradients = True

    # The subclass should implement this with its own negotiation strategy
    def _negotiate_grads_by_strategy(self, grads_and_vars):
        raise RuntimeError('Not implemented')

    def _set_num_gradients(self, n):
        raise RuntimeError('Not implemented')

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients and negotiate with peers."""
        grads_and_vars = self._optimizer.compute_gradients(*args, **kwargs)
        grads_and_vars_to_negotiate = []
        for grad, var in grads_and_vars:
            if grad is not None:
                grads_and_vars_to_negotiate.append((grad, var))

        def build_op():
            # returns negotiated (gradient, variable) pairs
            return self._negotiate_grads_by_strategy(
                grads_and_vars_to_negotiate)

        if self._enable_set_num_gradients:
            n_grads = len(grads_and_vars_to_negotiate)
            with tf.control_dependencies([self._set_num_gradients(n_grads)]):
                return build_op()
        else:
                return build_op()


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
