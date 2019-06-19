import tensorflow as tf
from kungfu.ops import barrier, broadcast, save_variables, adaptive_request_variables

from .core import KungFuOptimizer


class AdaptivePeerModelAveraging(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 window_size=10,
                 device_dense='',
                 device_sparse=''):
        super(AdaptivePeerModelAveraging, self).__init__(optimizer, name, use_locking,
                                                 device_dense, device_sparse)
        self.window_size = window_size

    @staticmethod
    def get_initializer():
        g = tf.get_default_graph()
        ops = []
        # TODO: auto inject tf.global_variables_initializer
        # with tf.control_dependencies([tf.global_variables_initializer()]):
        variables = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for v in variables:
            ops.append(tf.assign(v, broadcast(v)))
        with tf.control_dependencies(
                [save_variables(variables)]):
                return barrier()

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Calls this same method on the underlying optimizer."""

        grads, variables = zip(*grads_and_vars)

       
        other_peer_vars = adaptive_request_variables(variables, window_size=self.window_size)

        assign_ops = [
            tf.assign(v, 0.5 * (v + other_v))
            for ((g, v), other_v) in zip(grads_and_vars, other_peer_vars)
        ]

        apply_op = self._optimizer.apply_gradients(grads_and_vars,
                                                    **kwargs)
        save_model_op = save_variables(variables)

        with tf.control_dependencies(assign_ops):
            with tf.control_dependencies([apply_op]):
                with tf.control_dependencies([save_model_op]):
                    return tf.group(apply_op)


    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        return grads_and_vars_to_negotiate
