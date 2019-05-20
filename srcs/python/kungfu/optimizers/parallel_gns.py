from __future__ import print_function

import tensorflow as tf

from kungfu.ops import group_all_reduce, set_num_gradients, \
                       get_global_gradient_noise_operator, \
                       gradient_noise_controller
from .core import KungFuOptimizer


class ParallelOptimizerGNS(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse='',
                 use_global_step=True,
                 device_batch_size=None):
        super(ParallelOptimizerGNS, self).__init__(optimizer, name, use_locking,
                                                device_dense, device_sparse)
        self.device_batch_size = device_batch_size

    def _concat(self, ts):
        return tf.concat([tf.reshape(t, [-1]) for t in ts], -1)

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        """Negotiate grad with peers, following flexible strategy."""
        grads_to_negotiate = []
        variables_to_update = []
        for grad, var in grads_and_vars_to_negotiate:
            grads_to_negotiate.append(grad)
            variables_to_update.append(var)
        negotiated_grads = group_all_reduce(grads_to_negotiate)

        if self.device_batch_size is None:
            return list(zip(negotiated_grads, variables_to_update))
        else:
            noise_op = get_global_gradient_noise_operator(self.device_batch_size, 
                                                         self._concat(grads_to_negotiate),
                                                         self._concat(negotiated_grads))
            controller_future_batch_op = gradient_noise_controller(noise_op)
            with tf.control_dependencies([noise_op, controller_future_batch_op]):
                return list(zip([tf.identity(g) for g in negotiated_grads], variables_to_update))
        
    def _set_num_gradients(self, n):
        return set_num_gradients(tf.constant(n, tf.int32))
