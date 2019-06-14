import tensorflow as tf
from kungfu.internal import _get_other_ranks
from kungfu.ops import (broadcast, request_average_model, request_model,
                        save_model)

from .core import KungFuOptimizer


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
        self._request_mode = request_mode
        self._model_averaging_device = model_averaging_device
        self._peer_selection_strategy = peer_selection_strategy

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

    def _average_ops(self, variables):
        other_ranks = _get_other_ranks()
        if self._model_averaging_device == 'cpu':
            avg_vars = request_average_model(other_ranks, variables,
                                             self._request_mode,
                                             self._peer_selection_strategy)
            return [
                tf.assign(v, ave_v) for (v, ave_v) in zip(variables, avg_vars)
            ]
        elif self._model_averaging_device == 'gpu':
            other_peer_vars = request_model(other_ranks, variables,
                                            self._request_mode,
                                            self._peer_selection_strategy)
            return [
                tf.assign(v, 0.5 * (v + other_v))
                for (v, other_v) in zip(variables, other_peer_vars)
            ]
        else:
            raise Exception(
                "PeerModelAveraging optimizer does not support provided request model type."
            )

    def apply_gradients(self, grads_and_vars, **kwargs):
        variables = [v for g, v in grads_and_vars]
        with tf.control_dependencies(self._average_ops(variables)):
            # Calls this same method on the underlying optimizer.
            apply_op = self._optimizer.apply_gradients(grads_and_vars,
                                                       **kwargs)
            with tf.control_dependencies([apply_op]):
                with tf.control_dependencies([save_model(variables)]):
                    return tf.identity(apply_op)

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        return grads_and_vars_to_negotiate
