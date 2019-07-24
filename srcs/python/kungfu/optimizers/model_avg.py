import tensorflow as tf
from kungfu.internal import _get_num_peers, _get_other_ranks, _get_self_rank
from kungfu.ops import (barrier, broadcast, get_neighbour_mask,
                        get_peer_latencies, global_minimum_spanning_tree,
                        model_averaging, request, request_model, round_robin,
                        save_model, save_variables)

from .core import KungFuOptimizer


class ModelAveragingOptimizer(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""
    def __init__(self,
                 optimizer,
                 model_averaging_device="cpu",
                 request_mode="sync",
                 peer_selection_strategy="random",
                 name=None,
                 use_locking=False):
        super(ModelAveragingOptimizer, self).__init__(optimizer, name,
                                                      use_locking)
        self.request_mode = request_mode
        self.model_averaging_device = model_averaging_device
        self.peer_selection_strategy = peer_selection_strategy

    @staticmethod
    def get_initializer():
        # TODO: auto inject tf.global_variables_initializer
        # with tf.control_dependencies([tf.global_variables_initializer()]):
        ops = []
        # variables = tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        variables = tf.trainable_variables()
        for v in variables:
            ops.append(tf.assign(v, broadcast(v)))
        with tf.control_dependencies(ops):
            with tf.control_dependencies([save_model(variables)]):
                return barrier()

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


class AdaptiveModelAveragingOptimizer(KungFuOptimizer):
    """An optimizer that changes the topology dynamically."""
    def __init__(self, optimizer, name=None, use_locking=False, schedule=5):
        super(AdaptiveModelAveragingOptimizer,
              self).__init__(optimizer, name, use_locking)

        self._schedule = tf.constant(schedule, dtype=tf.int64)
        self._alpha = 0.5

        np = _get_num_peers()
        rank = _get_self_rank()

        local_step = tf.Variable(tf.zeros([], dtype=tf.int64), trainable=False)
        inc_local_step = tf.assign_add(local_step, 1)

        init_mask = tf.constant([r != rank for r in range(np)])
        neighbour_mask = tf.Variable(init_mask, trainable=False)

        def _update_mask():
            latencies = get_peer_latencies(local_step)
            mst_edges = global_minimum_spanning_tree(latencies)
            new_mask = get_neighbour_mask(mst_edges)
            return tf.assign(neighbour_mask, new_mask)

        with tf.control_dependencies([inc_local_step]):
            self._adapt_op = tf.cond(
                tf.equal(tf.mod(local_step, self._schedule), 0), _update_mask,
                tf.no_op)

        self._target = round_robin(neighbour_mask)

    def _average(self, v, other_v):
        return tf.assign(v, self._alpha * v + (1 - self._alpha) * other_v)

    @staticmethod
    def get_initializer():
        ops = []
        # variables = tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        variables = tf.trainable_variables()
        for v in variables:
            ops.append(tf.assign(v, broadcast(v)))
        with tf.control_dependencies(ops):
            with tf.control_dependencies([save_variables(variables)]):
                return barrier()

    def apply_gradients(self, grads_and_vars, **kwargs):
        _, variables = zip(*grads_and_vars)

        requested_vars = [request(self._target, v.name, v) for v in variables]

        average_ops = [
            self._average(v, other_v)
            for (v, other_v) in zip(variables, requested_vars)
        ]

        save_ops = save_variables(variables)

        apply_op = self._optimizer.apply_gradients(grads_and_vars, **kwargs)

        with tf.control_dependencies([self._adapt_op]):
            with tf.control_dependencies(average_ops):
                with tf.control_dependencies([apply_op]):
                    with tf.control_dependencies([save_ops]):
                        return tf.group(apply_op)

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        return grads_and_vars_to_negotiate
