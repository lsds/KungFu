import tensorflow as tf

from kungfu.ops import global_step_modifier, all_reduce, set_num_gradients
from .core import KungFuOptimizer


class ParallelOptimizer(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse='',
                 use_global_step=True):
        super(ParallelOptimizer, self).__init__(optimizer, name, use_locking,
                                                device_dense, device_sparse)

        pass

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        """Negotiate grad with peers, following flexible strategy."""

        def build_op():
            negotiated_grad_and_vars = []
            for grad, var in grads_and_vars_to_negotiate:
                with tf.variable_scope('NegotiatedGrad'):
                    negotiated_grad_and_vars.append((all_reduce(grad), var))
            return negotiated_grad_and_vars

        if self._use_global_step:
            with tf.control_dependencies([self._modify_trained_steps]):
                return build_op()
        else:
            return build_op()
