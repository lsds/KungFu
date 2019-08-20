from .core import KungFuOptimizer
from kungfu.ops import broadcast, get_init_version, group_all_reduce


class DynamicOptimizer(KungFuOptimizer):
    def __init__(self, optimizer, name=None, use_locking=False):
        super(DynamicOptimizer, self).__init__(optimizer, name, use_locking)

    @staticmethod
    def get_initializer():
        import tensorflow as tf

        init_version = get_init_version()
        if init_version == 0:
            tf_init = tf.global_variables_initializer()
        else:  # > 0
            tf_init = tf.no_op()

        # for all versions, including 0
        with tf.control_dependencies([tf_init]):
            bcast_ops = []
            variables = tf.trainable_variables()
            for v in variables:
                bcast_ops.append(tf.assign(v, broadcast(v)))
            return tf.group(bcast_ops)

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients and negotiate with peers."""
        grads_and_vars = self._optimizer.compute_gradients(*args, **kwargs)
        grads_and_vars_to_negotiate = [(g, v) for g, v in grads_and_vars
                                       if g is not None]
        return self._negotiate_grads_by_strategy(grads_and_vars_to_negotiate)

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        """Negotiate grads with current peers, using plain allreduce."""
        gradients, variables = list(zip(*grads_and_vars_to_negotiate))
        negotiated_gradients = group_all_reduce(gradients)
        return list(zip(negotiated_gradients, variables))

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Calls this same method on the underlying optimizer."""
        apply_op = self._optimizer.apply_gradients(grads_and_vars, **kwargs)
        return apply_op

    # def minimize(self, loss):
    #     pass
