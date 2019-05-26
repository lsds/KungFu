import tensorflow as tf

from kungfu.ops import request_vars, _tensor_size, _concat
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

    def avg(self, my_var, other_var):
        add_op = tf.add(my_var, other_var)
        avg_op = 0.5 * add_op
        return avg_op

    def model_average(self, my_vars, other_peer_vars):
        update_ops = []
        for my_v, other_v in zip(my_vars, other_peer_vars):
            with tf.device(my_v.device):
                avg = self.avg(my_v.read_value(), other_v.read_value())
                update_ops.append(tf.assign(my_var, avg))
        return update_ops

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Calls this same method on the underlying optimizer."""
        
        grads, variables = zip(*grads_and_vars)

        other_peer_vars = request_vars([i for i in range(_get_num_peers())], variables)

        # Make configurable momentum
        momentum = 0
        if isinstance(self, tf.train.MomentumOptimizer):
            momentum = self._optimizer._momentum
            
        assign_ops = [tf.assign(v, v + momentum * (v - other_v)) for ((g, v), other_v) in zip(grads_and_vars, other_peer_vars)]
        with tf.control_dependencies(assign_ops):
            # Check if assignment takes place
            return self._optimizer.apply_gradients(grads_and_vars, **kwargs)  

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        return grads_and_vars_to_negotiate
