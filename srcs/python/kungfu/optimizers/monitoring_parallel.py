import tensorflow as tf

from .core import KungFuOptimizer, lazy_load_op_lib

class MonitoringParallelOptimizer(KungFuOptimizer):
    """An optimizer that reduce gradients for synchronisation and compute the varience of gradients for monitoring."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse='',
                 use_global_step=True):
        super(MonitoringParallelOptimizer, self).__init__(optimizer, name, use_locking,
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

        with tf.variable_scope('NegotiatedGrad'):
            with tf.control_dependencies([self._op_lib.global_variance(grad)]):
                return self._op_lib.all_reduce(grad)

    def _set_num_gradients(self, n):
        return self._op_lib.set_num_gradients(tf.constant(n, tf.int32))