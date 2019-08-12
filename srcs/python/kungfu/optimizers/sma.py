import tensorflow as tf
from kungfu.ops import group_all_reduce
from kungfu.internal import _get_num_peers

from .core import KungFuOptimizer


class SynchronousModelAveragingOptimizer(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""
    def __init__(self, optimizer, name=None, use_locking=False, mu=0.5):
        super(SynchronousModelAveragingOptimizer, self).__init__(optimizer, name, use_locking)

        self.alpha = 1.0 / float(_get_num_peers())
        self.mu = mu

        self.prev_global_average_variables = [
            tf.Variable(tf.zeros(shape=v.shape, dtype=tf.float32), trainable=False) for v in tf.trainable_variables()
        ]

        self.global_average_variables = [
            tf.Variable(tf.zeros(shape=v.shape, dtype=tf.float32), trainable=False) for v in tf.trainable_variables()
        ]

        self.z_prime = [
            tf.Variable(tf.zeros(shape=z.shape, dtype=tf.float32), trainable=False) for z in self.global_average_variables
        ]


    @staticmethod
    def get_initializer():
        # TODO: auto inject tf.global_variables_initializer
        # with tf.control_dependencies([tf.global_variables_initializer()]):
        ops = []
        ops_global_model = []
        # variables = tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        variables = tf.trainable_variables()
        for _, i in enumerate(variables):
            ops.append(tf.assign(variables[i], broadcast(variables[i])))
            ops_global_model.append(tf.assign(self.global_average_variables[i], variables[i]))

        with tf.control_dependencies(ops):
            with tf.control_dependencies(ops_global_model):
                with tf.control_dependencies([save_model(variables)]):
                    return barrier()

    def apply_gradients(self, grads_and_vars, **kwargs):
        gradients, variables = zip(*grads_and_vars)

        # c_j
        gradient_corrections = [self.alpha * (v -  z) for (v, z) in zip(variables, self.global_average_variables)]
        corrected_gradients = [g - c for (g, c) in zip(gradients, gradient_corrections)]

        corrections_sum = group_all_reduce(gradient_corrections)


        z_prime_assign_ops = [tf.assign(z_prime_var, z) for (z_prime_var, z) in zip(self.z_prime, self.global_average_variables)]

        update_global_average_model = [tf.assign(z, z + corrections_sum + self.mu * (z - z_prev)) 
                    for (z, z_prev) in zip(self.global_average_variables, self.prev_global_average_variables)
        ]

        z_prev_assign_ops = [tf.assign(z_prev_var, z_prime_var) for (z_prime_var, z_prev_var) in zip(self.z_prime, self.prev_global_average_variables)]


        with tf.control_dependencies(corrections_sum):
            with tf.control_dependencies(z_prime_assign_ops):
                with tf.control_dependencies(update_global_average_model):
                    with tf.control_dependencies(z_prev_assign_ops):
                         return self._optimizer.apply_gradients(list(zip(corrected_gradients, variables)), **kwargs)


    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
            return grads_and_vars_to_negotiate
