import tensorflow as tf
from kungfu.tensorflow.compat import _tf_assign
from kungfu.tensorflow.ops import current_cluster_size, group_all_reduce

from .core import (_create_kungfu_keras_optimizer, _create_kungfu_optimizer,
                   _KungFuAlgorithm)


def SynchronousAveragingOptimizer(optimizer,
                                  name=None,
                                  alpha=0.1,
                                  use_locking=False,
                                  with_keras=False):
    """SynchronousAveragingOptimizer implements the [SMA]_ algorithm.

    [EA-SGD]_ proposed to use model averaging to train deep learning models and prove its convergence.
    [SMA]_ further improves [EA-SGD]_ results and show model averaging can benefit small-batch training
    and achieves fast convergence compared to synchronous SGD.

    .. [EA-SGD] Deep learning with Elastic Averaging SGD, NIPS 2015, `EA-SGD Paper <https://arxiv.org/abs/1412.6651>`_
    .. [SMA] CrossBow: Scaling Deep Learning with Small Batch Sizes on Multi-GPU Servers, VLDB 2019, `SMA Paper <http://www.vldb.org/pvldb/vol12/p1399-koliousis.pdf>`_

    Arguments:
        optimizer {tf.train.Optimizer, tf.keras.optimizers.Optimizer} -- Optimizer to use for computing gradients and applying updates.

    Keyword Arguments:
        - name {str} -- name prefix for the operations created when applying gradients. Defaults to "KungFu" followed by the provided optimizer type. (default: {None})
        - alpha {float} -- the ratio of a central model during averaging (Check the SMA and EA-SGD papers for its intuition). (default: {0.1})
        - use_locking {bool} -- Whether to use locking when updating variables. (default: {False})
        - with_keras {bool} -- Runs with pure Keras or not (default: {False})

    Raises:
        TypeError: Wrapped optimizer is not a subclass of tf.train.Optimizer or tf.keras.optimizers.Optimizer

    Returns:
        optimizer {tf.train.Optimizer, tf.keras.optimizers.Optimizer} -- KungFu distributed optimizer
    """
    sma_algo = _SynchronousAveraging(alpha)
    if not with_keras:
        return _create_kungfu_optimizer(optimizer, sma_algo, name, use_locking)
    else:
        return _create_kungfu_keras_optimizer(optimizer, sma_algo)


class _SynchronousAveraging(_KungFuAlgorithm):
    def __init__(self, alpha):
        self._num_workers = current_cluster_size()
        self._alpha = alpha

    def apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        gradients, variables = list(zip(*grads_and_vars))

        # filter out grad == None
        filtered_variables = [
            var for (grad, var) in list(zip(gradients, variables))
            if grad is not None
        ]

        # It is important to apply model averaging every iteration [2]
        sum_vars = group_all_reduce(filtered_variables)
        avg_vars = [g / self._num_workers for g in sum_vars]

        # TODO: Apply momentum to the averaged model [2]
        assign_ops = [
            _tf_assign(v, (1 - self._alpha) * v + self._alpha * avg_v)
            for v, avg_v in zip(filtered_variables, avg_vars)
        ]

        # We need to re-zip gradients and variables as grads_and_vars can be only unzipped once.
        new_grads_and_vars = zip(gradients, variables)

        # We can overlap model averaging and local SGD [2].
        with tf.control_dependencies(assign_ops):
            return apply_grads_func(new_grads_and_vars, **kwargs)
