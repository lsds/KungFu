import tensorflow as tf
from kungfu.tensorflow.v1.ops import (barrier, broadcast, current_cluster_size,
                                      current_rank, request_variable,
                                      request_variable_with_template,
                                      save_variable)

from .core import KungFuOptimizer, defuse, fuse


def get_random_peer(cluster_size, self_rank):
    t = tf.random_uniform([], minval=0, maxval=cluster_size, dtype=tf.int32)
    return tf.cond(tf.equal(t, self_rank), lambda: tf.mod(t + 1, cluster_size),
                   lambda: tf.identity(t))


class PairAveragingOptimizer(KungFuOptimizer):
    """PairAveragingOptimizer implements communication-efficient asynchronous training.

    Every iteration of training, this optimizer
    (1) Randomly selects a peer in the current cluster.
    (2) Pulls the selected peer's model
    (3) Performs model averaging with the local model.
    (4) Applies local gradients
    (5) Saves the model to a local store which allows other peers to pull from.

    This optimizer realizes the principle proposed in the following paper:
    Asynchronous Decentralized Parallel Stochastic Gradient Descent, ICML 2018
    https://arxiv.org/abs/1710.06952

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      device_batch_size:
        The training batch size of the local device
      fuse_variables:
        Fusing variables before saving a model.
        Turning it off to overlap training and synchronization.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "KungFu" followed by the provided
        optimizer type.
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.

    """
    def __init__(self,
                 optimizer,
                 fuse_variables=True,
                 name=None,
                 use_locking=False):
        super(PairAveragingOptimizer, self).__init__(optimizer, name,
                                                     use_locking)
        self._fuse_variables = fuse_variables
        self._step = tf.Variable(0, trainable=False, dtype=tf.int32)

    def _build_request_ops(self, target, variables):
        if self._fuse_variables:
            var_fused = fuse(variables)
            other_peer_var_fused = request_variable(target,
                                                    version=None,
                                                    name='KUNGFU_MODEL',
                                                    shape=var_fused.shape,
                                                    dtype=var_fused.dtype)
            return defuse(other_peer_var_fused, [v.shape for v in variables])
        else:
            return [
                request_variable_with_template(target, v) for v in variables
            ]

    def _build_save_op(self, variables):
        if self._fuse_variables:
            var_fused = fuse(variables)
            return save_variable(var_fused, name='KUNGFU_MODEL')
        else:
            return tf.group([save_variable(v) for v in variables])

    def compute_gradients(self, *args, **kwargs):
        self._sync_state_op = tf.cond(tf.equal(self._step, 0),
                                      self.distributed_initializer, tf.no_op)
        with tf.control_dependencies([self._sync_state_op]):
            with tf.control_dependencies([tf.assign_add(self._step, 1)]):
                return self._optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Calls this same method on the underlying optimizer."""
        np, rank = current_cluster_size(), current_rank()
        target = get_random_peer(np, rank)
        variables = [v for _g, v in grads_and_vars]

        with tf.control_dependencies([self._sync_state_op]):
            other_peer_vars = self._build_request_ops(target, variables)

        save_model_op = self._build_save_op(variables)

        assign_ops = [
            tf.assign(v, 0.5 * (v + other_v))
            for v, other_v in zip(variables, other_peer_vars)
        ]

        apply_op = self._optimizer.apply_gradients(grads_and_vars, **kwargs)

        with tf.control_dependencies(assign_ops):
            with tf.control_dependencies([apply_op]):
                with tf.control_dependencies([save_model_op]):
                    return tf.group(apply_op)

    def distributed_initializer(self):
        bcast_ops = []
        for v in self.variables():
            bcast_ops.append(tf.assign(v, broadcast(v)))

        variables = tf.trainable_variables()
        with tf.control_dependencies(bcast_ops):
            with tf.control_dependencies([self._build_save_op(variables)]):
                return barrier()
