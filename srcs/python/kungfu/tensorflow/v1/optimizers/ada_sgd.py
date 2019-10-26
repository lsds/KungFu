import tensorflow as tf
from kungfu.tensorflow.v1.ops import (barrier, broadcast, current_cluster_size,
                                      current_rank, group_all_reduce,
                                      request_variable_with_template,
                                      save_variable)

from .async_sgd import get_random_peer
from .core import KungFuOptimizer, defuse, fuse


class AdaptiveSGDOptimizer(KungFuOptimizer):
    """AdaptiveSGDOptimizer.

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "KungFuOptimizer" followed by the provided
        optimizer type.
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.

    """
    def __init__(self, optimizer, interval, name=None, use_locking=False):
        super(AdaptiveSGDOptimizer, self).__init__(optimizer,
                                                   name,
                                                   use_locking=use_locking)
        self._num_workers = current_cluster_size()
        self._rank = current_rank()
        self._step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._interval = interval

    def _build_request_and_save_ops(self, target, variables):
        var_fused = fuse(variables)
        save_model_op = save_variable(var_fused)
        other_peer_var_fused = request_variable_with_template(
            target, var_fused)
        other_peer_vars = defuse(other_peer_var_fused,
                                 [v.shape for v in variables])
        self._save_model_op = save_model_op  # save for _get_initializer_op
        return other_peer_vars, save_model_op

    # Asynchronous decentralised parallel SGD
    def _async_ma_sgd(self, grads_and_vars, **kwargs):
        target = get_random_peer(self._num_workers, self._rank)
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

    # Synchronous model averaging SGD (SMA)
    def _sync_ma_sgd(self, grads_and_vars, **kwargs):
        _, variables = list(zip(*grads_and_vars))
        sum_vars = group_all_reduce(variables)
        avg_vars = [g / self._num_workers for g in sum_vars]
        assign_ops = [
            tf.assign(v, avg_v) for v, avg_v in zip(variables, avg_vars)
        ]

        with tf.control_dependencies(assign_ops):
            return self._optimizer.apply_gradients(grads_and_vars, **kwargs)

    def apply_gradients(self, grads_and_vars, **kwargs):
        cond_op = tf.equal(tf.mod(self._step, self._interval), 0)
        with tf.control_dependencies([tf.assign_add(self._step, 1)]):
            return tf.cond(
                cond_op, lambda: self._sync_ma_sgd(grads_and_vars, **kwargs),
                lambda: self._async_ma_sgd(grads_and_vars, **kwargs))

    def distributed_initializer(self):
        bcast_ops = []
        for v in self.variables():
            bcast_ops.append(tf.assign(v, broadcast(v)))

        with tf.control_dependencies(bcast_ops):
            with tf.control_dependencies([self._save_model_op]):
                return barrier()
