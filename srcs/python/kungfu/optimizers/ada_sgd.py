import tensorflow as tf
from kungfu.ops import (broadcast, group_all_reduce, peer_info)

from .core import KungFuOptimizer, defuse, fuse


class AdaptiveSGDOptimizer(KungFuOptimizer):
    """AdaptiveSGDOptimizer.

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "Distributed" followed by the provided
        optimizer type.
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.

    """
    def __init__(self, optimizer, interval, name=None, use_locking=False):
        super(AdaptiveSGDOptimizer, self).__init__(optimizer,
                                                   name,
                                                   use_locking=use_locking)
        _rank, np = peer_info()
        # FIXME: use type of gradient
        self._num_workers = tf.cast(np, tf.float32)
        self._step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._interval = interval

    def _build_request_and_save_ops(self, target, variables):
        if self._fuse_variables:
            var_fused = fuse(variables)
            save_model_op = save_variable(var_fused)
            other_peer_var_fused = request_variable_with_template(
                target, var_fused)
            other_peer_vars = defuse(other_peer_var_fused,
                                     [v.shape for v in variables])
        else:
            save_model_op = tf.group([save_variable(v) for v in variables])
            other_peer_vars = [
                request_variable_with_template(target, v) for v in variables
            ]
        self._save_model_op = save_model_op  # save for _get_initializer_op
        return other_peer_vars, save_model_op

    def _async_sgd(self, grads_and_vars, **kwargs):
        np, rank = current_cluster_size(), current_rank()
        target = get_random_peer(np, rank)
        variables = [v for _g, v in grads_and_vars]
        other_peer_vars, save_model_op = self._build_request_and_save_ops(
            target, variables)

        assign_ops = [
            tf.assign(v, 0.5 * (v + other_v))
            for v, other_v in zip(variables, other_peer_vars)
        ]

        apply_op = self._optimizer.apply_gradients(grads_and_vars, **kwargs)

        with tf.control_dependencies(assign_ops):
            with tf.control_dependencies([apply_op]):
                with tf.control_dependencies([save_model_op]):
                    return tf.group(apply_op)

    def _sync_sgd(self, grads_and_vars, **kwargs):
        gradients, variables = list(zip(*grads_and_vars))
        summed_gradients = group_all_reduce(gradients)
        reduced_grads = [g / self._num_workers for g in summed_gradients]
        grads_and_vars = zip(reduced_grads, variables)
        apply_op = self._optimizer.apply_gradients(grads_and_vars, **kwargs)
        return apply_op

    def apply_gradients(self, grads_and_vars, **kwargs):
        # Adaptation logic
        adapt_op = tf.no_op()
        cond_op = tf.equal(tf.mod(self._step, self._interval), 0)
        with tf.control_dependencies([adapt_op]):
            return tf.cond(cond_op,
                           lambda: self._sync_sgd(grads_and_vars, **kwargs),
                           lambda: self._async_sgd(grads_and_vars, **kwargs))

    def distributed_initializer(self):
        bcast_ops = []
        for v in self.variables():
            bcast_ops.append(tf.assign(v, broadcast(v)))

        with tf.control_dependencies(bcast_ops):
            with tf.control_dependencies([self._save_model_op]):
                return barrier()
