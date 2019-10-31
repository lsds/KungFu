import tensorflow as tf
from kungfu.tensorflow import _tf_assign, _tf_mod
from kungfu.tensorflow.v1.ops import (barrier, broadcast, counter,
                                      current_cluster_size, current_rank,
                                      request_variable,
                                      request_variable_with_template,
                                      save_variable)

from .core import KungFuOptimizer, defuse, fuse


def get_random_peer(cluster_size, self_rank):
    t = tf.random.uniform([], minval=0, maxval=cluster_size, dtype=tf.int32)
    return tf.cond(tf.equal(t,
                            self_rank), lambda: _tf_mod(t + 1, cluster_size),
                   lambda: tf.identity(t))


class PairAveragingOptimizer(KungFuOptimizer):
    """PairAveragingOptimizer implements the [AD-PSGD]_ algorithm.

    Every iteration of training, this optimizer:

    1. Randomly selects a peer in the current cluster.
    2. Pulls the selected peer's model
    3. Performs model averaging with the local model.
    4. Applies local gradients
    5. Saves the model to a local store which allows other peers to pull from.

    .. [AD-PSGD] Asynchronous Decentralized Parallel Stochastic Gradient Descent, ICML 2018, `AD-PSGD Paper <https://arxiv.org/abs/1710.06952>`_

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      fuse_requests:
        Fusing the requests for remote variables to amortise communication cost.
        The fusing however takes extra memory.
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
                 fuse_requests=True,
                 name=None,
                 use_locking=False):
        super(PairAveragingOptimizer, self).__init__(optimizer, name,
                                                     use_locking)
        self._fuse_requests = fuse_requests
        self._step = counter()
        self._fused_model_name = optimizer.get_name()

    def _build_request_ops(self, target, variables):
        if self._fuse_requests:
            var_fused = fuse(variables)
            other_peer_var_fused = request_variable(
                target,
                version=None,
                name=self._fused_model_name,
                shape=var_fused.shape,
                dtype=var_fused.dtype)
            return defuse(other_peer_var_fused, [v.shape for v in variables])
        else:
            return [
                request_variable_with_template(target, v) for v in variables
            ]

    def _build_save_op(self, variables):
        if self._fuse_requests:
            var_fused = fuse(variables)
            return save_variable(var_fused, name=self._fused_model_name)
        else:
            return tf.group([save_variable(v) for v in variables])

    def init_store(self, variables):
        with tf.control_dependencies([self._build_save_op(variables)]):
            return barrier()

    def apply_gradients(self, grads_and_vars, **kwargs):
        np, rank = current_cluster_size(), current_rank()
        target = get_random_peer(np, rank)
        gradients, variables = list(zip(*grads_and_vars))

        init_store_op = tf.cond(tf.equal(self._step, 0),
                                lambda: self.init_store(variables), tf.no_op)
        with tf.control_dependencies([init_store_op]):
            other_peer_vars = self._build_request_ops(target, variables)

        save_model_op = self._build_save_op(variables)

        assign_ops = [
            _tf_assign(v, 0.5 * (v + other_v))
            for v, other_v in zip(variables, other_peer_vars)
        ]

        # This is important to re-zip to ensure AutoGraph can track the dependency
        new_grads_and_vars = zip(gradients, variables)
        apply_op = self._optimizer.apply_gradients(new_grads_and_vars,
                                                   **kwargs)

        with tf.control_dependencies(assign_ops):
            with tf.control_dependencies([apply_op]):
                with tf.control_dependencies([save_model_op]):
                    return tf.group(apply_op)
