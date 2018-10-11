import platform
import tensorflow as tf


class NegotiableOptimizer(tf.train.Optimizer):
    """An optimizer that would negotiate the gradients before apply it."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse=''):

        self._optimizer = optimizer
        self._device_dense = device_dense
        self._device_sparse = device_sparse
        if name is None:
            name = "NegotiableOptimizer{}".format(type(optimizer).__name__)
        super(NegotiableOptimizer, self).__init__(
            name=name, use_locking=use_locking)

    def _negotiate_grad(self, grad):
        raise RuntimeError('Not implemented')
        # The subclass should implement this with its own negotiation strategy

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients and negotiate with peers."""
        grads_and_vars = self._optimizer.compute_gradients(*args, **kwargs)
        negotiated_grad_and_vars = []
        for grad, var in grads_and_vars:
            if grad is not None:
                negotiated_grad_and_vars.append((self._negotiate_grad(grad),
                                                 var))
        return negotiated_grad_and_vars

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


def _load_op_lib(name):
    suffix = 'so' if platform.uname()[0] != 'Darwin' else 'dylib'
    return tf.load_op_library('./lib/lib%s.%s' % (name, suffix))


_op_lib = _load_op_lib('negotiator')

negotiator = _op_lib.negotiator

# TODO: tf.ops.NotDifferentiable('negotiator')


class AsyncSGDOptimizer(NegotiableOptimizer):
    """An optimizer that negotiates using the negotiator operator."""

    def _negotiate_grad(self, grad):
        """Negotiate grad with peers."""
        with tf.variable_scope('NegotiatedGrad'):
            return negotiator(grad)
