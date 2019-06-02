import tensorflow as tf

from kungfu.ops import broadcast, save_model, request_model, model_averaging
from .core import KungFuOptimizer


def _get_self_rank():
    import os
    return int(os.getenv('KUNGFU_TEST_SELF_RANK'))


def _get_num_peers():
    import json, os
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    return len(cluster_spec['Peers'])


class PeerModelAveraging(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""

    def __init__(self,
                 optimizer,
                 model_averaging_device="cpu",
                 request_mode="sync",
                 peer_selection_strategy="random",
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse=''):
        super(PeerModelAveraging, self).__init__(optimizer, name, use_locking,
                                                 device_dense, device_sparse)
        self.request_mode = request_mode
        self.model_averaging_device = model_averaging_device
        self.peer_selection_strategy = peer_selection_strategy

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

        if self.model_averaging_device == 'cpu':
            apply_avg_model = model_averaging(
                [i for i in range(_get_num_peers())], variables,
                self.request_mode, self.peer_selection_strategy)

            apply_op = self._optimizer.apply_gradients(grads_and_vars,
                                                       **kwargs)
            save_model_op = save_model(variables)

            with tf.control_dependencies([apply_avg_model]):
                with tf.control_dependencies([apply_op]):
                    with tf.control_dependencies([save_model_op]):
                        return tf.group(apply_op)
        elif self.model_averaging_device == 'gpu':
            other_peer_vars = request_model(
                [i for i in range(_get_num_peers())], variables,
                self.request_mode, self.peer_selection_strategy)

            assign_ops = [
                tf.assign(v, 0.5 * (v + other_v))
                for ((g, v), other_v) in zip(grads_and_vars, other_peer_vars)
            ]

            apply_op = self._optimizer.apply_gradients(grads_and_vars,
                                                       **kwargs)
            save_model_op = save_model(variables)

            with tf.control_dependencies(assign_ops):
                with tf.control_dependencies([apply_op]):
                    with tf.control_dependencies([save_model_op]):
                        return tf.group(apply_op)
        else:
            raise Exception(
                "PeerModelAveraging optimizer does not support provided request model type."
            )

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        return grads_and_vars_to_negotiate
