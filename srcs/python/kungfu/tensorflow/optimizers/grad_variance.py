import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.tensorflow.ops import (counter, current_cluster_size, current_rank,
                                   group_all_reduce)

from .core import _create_kungfu_optimizer, _KungFuAlgorithm


def MonitorGradientVarianceOptimizer(optimizer,
                                     monitor_interval=1,
                                     name=None,
                                     use_locking=False):
    """MonitorGradientVarianceOptimizer monitors gradient variance of synchronous SGD.

    You can find the defintion of variance of tensors here:
    https://en.wikipedia.org/wiki/Variance

    Arguments:
        optimizer {tf.train.Optimizer, tf.keras.optimizers.Optimizer} -- Optimizer to use for computing gradients and applying updates.

    Keyword Arguments:
        - monitor_interval {int} -- monitoring interval. (default: {1})
        - name {str} -- name prefix for the operations created when applying gradients. Defaults to "KungFu" followed by the provided optimizer type. (default: {None})
        - use_locking {bool} -- Whether to use locking when updating variables. (default: {False})

    Raises:
        TypeError: Wrapped optimizer is not a subclass of tf.train.Optimizer or tf.keras.optimizers.Optimizer

    Returns:
        optimizer {tf.train.Optimizer, tf.keras.optimizers.Optimizer} -- KungFu distributed optimizer
    """
    grad_var_algo = _GradVariance(monitor_interval)
    return _create_kungfu_optimizer(optimizer, grad_var_algo, name,
                                    use_locking)


class _GradVariance(_KungFuAlgorithm):
    def __init__(self, monitor_interval=1):
        self._num_workers = current_cluster_size()
        self._step = counter()

        self._interval = monitor_interval
        self._summed_variance = None
        self._variances = None

    def _monitor(self, grads, reduced_grads):
        square_grads = [tf.square(g) for g in grads]
        summed_square_grads = group_all_reduce(square_grads)
        reduced_square_grads = map_maybe(lambda g: g / self._num_workers,
                                         summed_square_grads)
        grad_variances = [
            square_grad - tf.square(grad)
            for square_grad, grad in zip(reduced_square_grads, reduced_grads)
        ]
        variances = [
            tf.norm(grad_variance) for grad_variance in grad_variances
        ]
        summed_variance = tf.reduce_sum(variances)
        return tf.print('Variance:', summed_variance)

    def apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        grads, variables = list(zip(*grads_and_vars))

        # Synchronization logic
        summed_grads = group_all_reduce(grads)
        reduced_grads = map_maybe(lambda g: g / self._num_workers,
                                  summed_grads)

        # Monitoring logic
        monitor_grads_op = tf.cond(
            tf.equal(tf.mod(self._step, self._interval), 0),
            lambda: self._monitor(grads, reduced_grads), lambda: tf.no_op())

        with tf.control_dependencies([monitor_grads_op]):
            return apply_grads_func(zip(reduced_grads, variables), **kwargs)
