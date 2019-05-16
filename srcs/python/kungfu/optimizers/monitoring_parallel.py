import tensorflow as tf

from kungfu.ops import all_reduce, global_variance
from .core import KungFuOptimizer


class MonitoringParallelOptimizer(KungFuOptimizer):
    """An optimizer that reduce gradients for synchronisation and compute the varience of gradients for monitoring."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse='',
                 use_global_step=True):
        super(MonitoringParallelOptimizer,
              self).__init__(optimizer, name, use_locking, device_dense,
                             device_sparse)

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        """Negotiate grad with peers, following flexible strategy."""

        negotiated_grad_and_vars = []
        for grad, var in grads_and_vars_to_negotiate:
            with tf.variable_scope('NegotiatedGrad'):
                with tf.control_dependencies([global_variance(grad)]):
                    negotiated_grad_and_vars.append((all_reduce(grad), var))
        return negotiated_grad_and_vars
