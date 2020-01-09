import tensorflow as tf
from kungfu.tensorflow.compat import _tf_assign, _tf_mod
from kungfu.tensorflow.ops import (barrier, counter, current_cluster_size,
                                   current_rank, defuse, fuse,
                                   request_variable,
                                   request_variable_with_template,
                                   save_variable)

from .core import (_create_kungfu_keras_optimizer, _create_kungfu_optimizer,
                   _KungFuAlgorithm)


def PairAveragingOptimizer(optimizer,
                           fuse_requests=True,
                           fused_model_name=None,
                           name=None,
                           use_locking=False,
                           with_keras=False):
    """PairAveragingOptimizer implements the [AD-PSGD]_ algorithm.

    Every iteration of training, this optimizer:

    1. Randomly selects a peer in the current cluster.
    2. Pulls the selected peer's model
    3. Performs model averaging with the local model.
    4. Applies local gradients
    5. Saves the model to a local store which allows other peers to pull from.

    .. [AD-PSGD] Asynchronous Decentralized Parallel Stochastic Gradient Descent, ICML 2018, `AD-PSGD Paper <https://arxiv.org/abs/1710.06952>`_

    Arguments:
        optimizer {tf.train.Optimizer, tf.keras.optimizers.Optimizer} -- Optimizer to use for computing gradients and applying updates.

    Keyword Arguments:
        - fuse_requests {bool} -- Fusing requests to amortise communication cost at the cost of extra GPU memory and cycles. (default: {True})
        - fused_model_name {str} -- The unique name for the fused model kept in a local store. (default: {None})
        - name {str} -- name prefix for the operations created when applying gradients. Defaults to "KungFu" followed by the provided optimizer type. (default: {None})
        - use_locking {bool} -- Whether to use locking when updating variables. (default: {False})
        - with_keras {bool} -- Runs with pure Keras or not (default: {False})

    Raises:
        TypeError: Wrapped optimizer is not a subclass of tf.train.Optimizer or tf.keras.optimizers.Optimizer

    Returns:
        optimizer {tf.train.Optimizer, tf.keras.optimizers.Optimizer} -- KungFu distributed optimizer
    """

    if fused_model_name is None:
        if hasattr(optimizer, 'get_name'):
            # tf.train.Optimizer
            fused_model_name = optimizer.get_name()
        else:
            try:
                # tf.keras.optimizers.Optimizer has name since tf1.15
                fused_model_name = optimizer.get_config()['name']
            except:
                # keras optimizer does not have name
                fused_model_name = 'PairAveragingOptimizer'
                print(
                    'WARNING: You must give a unique name if using parallel PairAveragingOptimizers.'
                )

    pair_avg = _PairAveraging(fuse_requests, fused_model_name=fused_model_name)

    if not with_keras:
        return _create_kungfu_optimizer(optimizer, pair_avg, name, use_locking)
    else:
        return _create_kungfu_keras_optimizer(optimizer, pair_avg)


def get_random_peer(cluster_size, self_rank):
    t = tf.random.uniform([], minval=0, maxval=cluster_size, dtype=tf.int32)
    return tf.cond(tf.equal(t,
                            self_rank), lambda: _tf_mod(t + 1, cluster_size),
                   lambda: tf.identity(t))


class _PairAveraging(_KungFuAlgorithm):
    def __init__(self, fuse_requests, fused_model_name=None):
        self._step = counter()
        self._fuse_requests = fuse_requests
        self._fused_model_name = fused_model_name

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

    def apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        np, rank = current_cluster_size(), current_rank()
        target = get_random_peer(np, rank)
        gradients, variables = list(zip(*grads_and_vars))

        # filter out grad == None
        filtered_variables = [
            var for (grad, var) in list(zip(gradients, variables))
            if grad is not None
        ]

        init_store_op = tf.cond(tf.equal(self._step, 0),
                                lambda: self.init_store(filtered_variables),
                                tf.no_op)
        with tf.control_dependencies([init_store_op]):
            other_peer_vars = self._build_request_ops(target,
                                                      filtered_variables)

        save_model_op = self._build_save_op(filtered_variables)

        assign_ops = [
            _tf_assign(v, 0.5 * (v + other_v))
            for v, other_v in zip(filtered_variables, other_peer_vars)
        ]

        # We need to re-zip gradients and variables as grads_and_vars can be only unzipped once.
        new_grads_and_vars = zip(gradients, variables)
        apply_op = apply_grads_func(new_grads_and_vars, **kwargs)

        with tf.control_dependencies(assign_ops):
            with tf.control_dependencies([apply_op]):
                with tf.control_dependencies([save_model_op]):
                    return tf.group(apply_op)
