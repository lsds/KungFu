import tensorflow as tf
from kungfu.ops import gpu_group_all_reduce


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

    def _negotiate_grad(self, grad):
        raise RuntimeError('Not implemented')
        # The subclass should implement this with its own negotiation strategy

    def _set_num_gradients(self, n):
        raise RuntimeError('Not implemented')

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients and negotiate with peers."""
        grads_and_vars = self._optimizer.compute_gradients(*args, **kwargs)
        grads_to_negotiate = []
        variables_to_update = []
        for grad, var in grads_and_vars:
            if grad is not None:
                grads_to_negotiate.append(grad)
                variables_to_update.append(var)
        # FIXME: auto determine device
        negotiated_grads = gpu_group_all_reduce(grads_to_negotiate)
        return list(zip(negotiated_grads, variables_to_update))

    # forward to the underlying optimizer

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
