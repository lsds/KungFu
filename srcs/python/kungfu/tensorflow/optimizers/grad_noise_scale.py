import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.tensorflow.ops import (counter, current_cluster_size, current_rank,
                                   all_reduce, fuse, global_noise_scale,
                                   group_all_reduce)

from .core import _create_kungfu_optimizer, _KungFuAlgorithm


def MonitorGradientNoiseScaleOptimizer(optimizer,
                                       device_batch_size,
                                       monitor_interval=1,
                                       alpha=0.6,
                                       name=None,
                                       use_locking=False):
    """MonitorGradientNoiseScaleOptimizer monitors the Gradient Noise Scale [GNS]_ of synchronous SGD.

     .. [GNS] An Empirical Model of Large-Batch Training, `GNS paper <https://arxiv.org/abs/1812.06162>`_

    Arguments:
        optimizer {tf.train.Optimizer, tf.keras.optimizers.Optimizer} -- Optimizer to use for computing gradients and applying updates.

    Keyword Arguments:
        - device_batch_size {int} -- the training batch size of the local device.
        - monitor_interval {int} -- monitoring interval. (default: {1})
        - name {str} -- name prefix for the operations created when applying gradients. Defaults to "KungFu" followed by the provided optimizer type. (default: {None})
        - use_locking {bool} -- Whether to use locking when updating variables. (default: {False})

    Raises:
        TypeError: Wrapped optimizer is not a subclass of tf.train.Optimizer or tf.keras.optimizers.Optimizer

    Returns:
        optimizer {tf.train.Optimizer, tf.keras.optimizers.Optimizer} -- KungFu distributed optimizer
    """
    mon_gns_algo = _GradientNoiseScale(device_batch_size, monitor_interval,
                                       alpha)
    return _create_kungfu_optimizer(optimizer, mon_gns_algo, name, use_locking)


class _GradientNoiseScale(_KungFuAlgorithm):
    def __init__(self, device_batch_size, monitor_interval, alpha):
        self._num_workers = current_cluster_size()
        self._step = counter()

        self._interval = monitor_interval
        self._device_batch_size = tf.cast(device_batch_size, dtype=tf.float32)
        self._global_batch_size = self._device_batch_size * self._num_workers
        self._alpha = alpha

        self._global_step = tf.train.get_or_create_global_step()
        self._gradient_noise_moving_average = tf.get_variable(
            "gradient_noise_moving_average", initializer=0.0, trainable=False)

    def _monitor(self, grads, reduced_grads):
        # Only the master node is doing the global monitoring.
        noise_op = global_noise_scale(self._device_batch_size,
                                      self._global_batch_size, fuse(grads),
                                      fuse(reduced_grads), self._alpha)

        abs_global_noise = tf.math.abs(noise_op)
        beta = 0.01

        gradient_noise_scale_ema = tf.cond(
            tf.math.equal(self._global_step, 0),  #
            lambda: tf.assign(self._gradient_noise_moving_average,
                              abs_global_noise),
            lambda: tf.assign(self._gradient_noise_moving_average,
                              (1 - beta) * self._gradient_noise_moving_average
                              + beta * abs_global_noise))

        sum_gradient_noise_scale_ema = all_reduce(gradient_noise_scale_ema)

        tf.summary.scalar('gradient_noise_scale', abs_global_noise)
        tf.summary.scalar('gradient_noise_scale_ema', gradient_noise_scale_ema)
        tf.summary.scalar('sum_gradient_noise_scale_ema',
                          sum_gradient_noise_scale_ema)
        print_op = tf.print('Gradient Noise Scale:', abs_global_noise)
        return print_op

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
