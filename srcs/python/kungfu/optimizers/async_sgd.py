import tensorflow as tf
from kungfu.internal import _get_num_peers, _get_other_ranks, _get_self_rank
from kungfu.ops import (_concat, barrier, broadcast, get_neighbour_mask,
                        get_peer_latencies, global_minimum_spanning_tree,
                        model_averaging, request, request_model, round_robin,
                        save_model, save_variable, save_variables)

from .core import KungFuOptimizer


def fuse(ts):
    return tf.concat([tf.reshape(t, [-1]) for t in ts], -1)


def defuse(y, shapes):
    ts = []
    off = 0
    for s in shapes:
        size = s.num_elements()
        x = tf.slice(y, [off], [size])
        x = tf.reshape(x, s)
        ts.append(x)
        off += size
    if off != y.shape.num_elements():
        raise RuntimeError('invalid dtype')
    return ts


def get_random_peer(cluster_size, self_rank):
    t = tf.random_uniform([], minval=0, maxval=cluster_size, dtype=tf.int32)
    return tf.cond(tf.equal(t, self_rank), lambda: tf.mod(t + 1, cluster_size),
                   lambda: tf.identity(t))


class ModelAveragingOptimizerNew(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""
    def __init__(self,
                 optimizer,
                 model_averaging_device="gpu",
                 request_mode="sync",
                 peer_selection_strategy="random",
                 name=None,
                 use_locking=False):
        super(ModelAveragingOptimizerNew,
              self).__init__(optimizer, name, use_locking)
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
            raise RuntimeError('NOT supported')
        elif self.model_averaging_device == 'gpu':
            shapes = [v.shape for v in variables]

            var_fused = fuse(variables)
            target = get_random_peer(_get_num_peers(), _get_self_rank())
            other_peer_var_fused = request(target, var_fused.name, var_fused)

            other_peer_vars = defuse(other_peer_var_fused, shapes)

            assign_ops = [
                tf.assign(v, 0.5 * (v + other_v))
                for ((g, v), other_v) in zip(grads_and_vars, other_peer_vars)
            ]

            apply_op = self._optimizer.apply_gradients(grads_and_vars,
                                                       **kwargs)
            save_model_op = save_variable(var_fused)

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
