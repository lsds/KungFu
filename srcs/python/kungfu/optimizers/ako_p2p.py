import tensorflow as tf

from kungfu.ops import broadcast, save_model, request_vars
from .core import KungFuOptimizer


def _get_self_rank():
    import os
    return int(os.getenv('KUNGFU_TEST_SELF_RANK'))


def _get_num_peers():
    import json, os
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    return len(cluster_spec['Peers'])


class AkoP2P(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse=''):
        super(AkoP2P, self).__init__(optimizer, name, use_locking,
                                     device_dense, device_sparse)

    @staticmethod
    def get_initializer():
        g = tf.get_default_graph()
        ops = []
        # TODO: auto inject tf.global_variables_initializer
        # with tf.control_dependencies([tf.global_variables_initializer()]):
        variables = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for v in variables:
            ops.append(tf.assign(v, broadcast(v)))
        with tf.control_dependencies(ops):
             return save_model(tf.trainable_variables())

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Calls this same method on the underlying optimizer."""
        
        grads, variables = zip(*grads_and_vars)

        other_peer_vars = request_vars([i for i in range(_get_num_peers())], variables)

        # Make configurable momentum
        momentum = 0
        if isinstance(self._optimizer, tf.train.MomentumOptimizer):
            print("Using momentum optimizer")
            momentum = self._optimizer._momentum
        else:
            raise Exception("Optimizer is not an instance of tf.train.MomentumOptimizer")

        # print_ovs = [tf.Print(o_v, [o_v], message="Other peer variable ID " + str(i) + ": ") for i, o_v in enumerate(other_peer_vars)]
        # print_lvs = [tf.Print(v, [v], message="My variable ID " + str(i) + ": ") for i, (_, v) in enumerate(grads_and_vars)]

        assign_ops = [tf.assign(v, 0.5 * (v + other_v), use_locking=False) for ((g, v), other_v) in zip(grads_and_vars, other_peer_vars)]
        #assign_ops = [tf.assign_add(v, momentum * (v - other_v)) for ((g, v), other_v) in zip(grads_and_vars, other_peer_vars)]

        apply_op = self._optimizer.apply_gradients(grads_and_vars, **kwargs) 
        save_model_op = save_model(variables)

        with tf.control_dependencies(assign_ops):
            with tf.control_dependencies([apply_op]):
                with tf.control_dependencies([save_model_op]):
                    # figure out type of apply_ops
                    return tf.group(apply_op)

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        return grads_and_vars_to_negotiate
