import tensorflow as tf

from kungfu.ops import group_all_reduce, set_num_gradients
from .core import KungFuOptimizer


from kungfu.ops import get_gradient_noise_operators
from kungfu.ops import build_controller_op, gradient_noise_summaries, global_noise_summaries

class ParallelOptimizer(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse='',
                 use_global_step=True,
                 device_batch_size=None):
        super(ParallelOptimizer, self).__init__(optimizer, name, use_locking,
                                                device_dense, device_sparse)
        self.device_batch_size = device_batch_size

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
            noise_ops = get_gradient_noise_operators(self.device_batch_size, grads_to_negotiate, negotiated_grads)
            noise_ops = [tf.abs(op) for op in noise_ops]
            total = tf.reduce_sum(noise_ops)
            global_noise_summaries(total, tf.div(total, len(negotiated_grads)))

            total = tf.div(total, len(grads_and_vars_to_negotiate)) # Average noise scale
            print_op = tf.Print(total, [total], message="Total Gradient Noise at current iteration")

            gradient_noise_summaries(noise_ops, grads)
            merged = tf.summary.merge_all()

            with tf.control_dependencies(noise_ops + [merged]):
                return build_controller_op(list(zip(negotiated_grads, variables_to_update)))

    def _set_num_gradients(self, n):
        return set_num_gradients(tf.constant(n, tf.int32))
