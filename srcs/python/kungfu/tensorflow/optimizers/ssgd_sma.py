import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.tensorflow.ops import (current_cluster_size, defuse, fuse,
                                   group_all_reduce, group_nccl_all_reduce)
from .core import (_create_kungfu_keras_optimizer, _create_kungfu_optimizer,
                   _KungFuAlgorithm)

def SSGD_SMA(optimizer,
            name=None,
            alpha=1,
            use_locking=False,
            with_keras=False):
    
    ssgd_sma_algo = _SSGD_SMA(alpha)
    if not with_keras:
        return _create_kungfu_optimizer(optimizer, ssgd_sma_algo, name, use_locking)
    else:
        return _create_kungfu_keras_optimizer(optimizer, ssgd_sma_algo)


class _SSGD_SMA(_KungFuAlgorithm):
    def __init__(self, alpha):
        self._num_workers = current_cluster_size()
        self._alpha = alpha

    def _ssgd_apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        gradients, variables = list(zip(*grads_and_vars))

        summed_gradients = group_all_reduce(gradients)

        reduced_grads = map_maybe(lambda g: g / self._num_workers,
                                  summed_gradients)

        # We need to re-zip gradients and variables as grads_and_vars can be only unzipped once.
        reduced_grads_and_vars = zip(reduced_grads, variables)

        return apply_grads_func(reduced_grads_and_vars, **kwargs)

    def _sma_apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        # It is important to apply model averaging every iteration [2]
        gradients, variables = list(zip(*grads_and_vars))
        sum_vars = group_all_reduce(variables)
        avg_vars = [g / self._num_workers for g in sum_vars]

        # TODO: Apply momentum to the averaged model [2]
        assign_ops = [
            _tf_assign(v, (1 - self._alpha) * v + self._alpha * avg_v)
            for v, avg_v in zip(variables, avg_vars)
        ]

        # We need to re-zip gradients and variables as grads_and_vars can be only unzipped once.
        new_grads_and_vars = zip(gradients, variables)

        # We can overlap model averaging and local SGD [2].
        with tf.control_dependencies(assign_ops):
            return apply_grads_func(new_grads_and_vars, **kwargs)

    def apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        num_training_steps_switch = 800
        return tf.cond(tf.math.less(kwargs["global_step"], num_training_steps_switch),
                            self._ssgd_apply_gradients(apply_grads_func, grads_and_vars, **kwargs),
                            self._sma_apply_gradients(apply_grads_func, grads_and_vars, **kwargs))
