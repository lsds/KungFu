import os
import sysconfig

import tensorflow as tf

__all__ = [
    'SyncSGDOptimizer',
]


def _load_op_lib(name):
    module_path = os.path.dirname(__file__)
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    filename = os.path.join(module_path, name + suffix)
    return tf.load_op_library(filename)


_op_lib_name = 'kungfu_tensorflow_ops'
_op_lib = None


def lazy_load_op_lib():
    global _op_lib
    if _op_lib is None:
        _op_lib = _load_op_lib(_op_lib_name)
    return _op_lib


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

        self._enable_set_gradient_count = True

    def _negotiate_grad(self, grad):
        raise RuntimeError('Not implemented')
        # The subclass should implement this with its own negotiation strategy

    def _set_gradient_count(self, n):
        raise RuntimeError('Not implemented')

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients and negotiate with peers."""
        grads_and_vars = self._optimizer.compute_gradients(*args, **kwargs)
        grads_and_vars_to_negotiate = []
        for grad, var in grads_and_vars:
            if grad is not None:
                grads_and_vars_to_negotiate.append((grad, var))

        def build_op():
            negotiated_grad_and_vars = []
            for grad, var in grads_and_vars_to_negotiate:
                negotiated_grad_and_vars.append((self._negotiate_grad(grad),
                                                 var))
            return negotiated_grad_and_vars

        if self._enable_set_gradient_count:
            grad_count = len(grads_and_vars_to_negotiate)
            with tf.control_dependencies(
                [self._set_gradient_count(grad_count)]):
                return build_op()
        else:
            return build_op()

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


class SyncSGDOptimizer(KungFuOptimizer):
    """An optimizer that negotiates using the negotiator operator."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse='',
                 use_global_step=True):
        super(SyncSGDOptimizer, self).__init__(optimizer, name, use_locking,
                                               device_dense, device_sparse)

        self._op_lib = lazy_load_op_lib()

        self._use_global_step = use_global_step
        if self._use_global_step:
            self._trained_steps = tf.Variable(tf.zeros([], tf.int32))
            self._modify_trained_steps = tf.assign(
                self._trained_steps,
                self._op_lib.global_step_modifier(self._trained_steps))

    def _negotiate_grad(self, grad):
        """Negotiate grad with peers."""

        def build_op():
            with tf.variable_scope('NegotiatedGrad'):
                return self._op_lib.negotiator(grad)

        if self._use_global_step:
            with tf.control_dependencies([self._modify_trained_steps]):
                return build_op()
        else:
            return build_op()

    def _set_gradient_count(self, n):
        return self._op_lib.set_gradient_count(tf.constant(n, tf.int32))
