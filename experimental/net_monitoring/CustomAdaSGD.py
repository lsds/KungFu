import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.tensorflow.compat import _tf_assign, _tf_hook
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import (counter, current_cluster_size,
                                   group_all_reduce)
from kungfu.tensorflow.optimizers.core import (_create_kungfu_keras_optimizer,
                                               _create_kungfu_optimizer,
                                               _KungFuAlgorithm)


def CustomAdaSGDOptimizer(optimizer,
                         alpha=0.1,
                         name=None,
                         use_locking=False,
                         with_keras=False):

    algo = _CustomAdaSGD(alpha)
    if not with_keras:
        return _create_kungfu_optimizer(optimizer, algo, name, use_locking)
    else:
        return _create_kungfu_keras_optimizer(optimizer, algo)


class _CustomAdaSGD(_KungFuAlgorithm):
    def __init__(self, alpha):
        self._num_workers = current_cluster_size()
        self._alpha = alpha
        self._global_step = tf.train.get_or_create_global_step()
        self._cond_var_Ada_var = tf.Variable(False, trainable=False, name='cond_var_Ada')

    def _ssgd(self, apply_grads_func, gradients, variables, **kwargs):
        sum_grads = group_all_reduce(gradients)
        avg_grads = map_maybe(lambda g: g / self._num_workers, sum_grads)

        # print_op = tf.print("Inside SSGD")

        # We need to re-zip gradients and variables as grads_and_vars can be only unzipped once.
        grads_and_vars = zip(avg_grads, variables)

        return apply_grads_func(grads_and_vars, **kwargs)

    def _sma(self, apply_grads_func, gradients, variables, **kwargs):
        # It is important to apply model averaging every iteration [2]
        # print_op = tf.print("Inside SMA")

        sum_vars = group_all_reduce(variables)
        avg_vars = [v / self._num_workers for v in sum_vars]

        # TODO: Apply momentum to the averaged model [2]
        assign_ops = [
            _tf_assign(v, (1 - self._alpha) * v + self._alpha * avg_v)
            for v, avg_v in zip(variables, avg_vars)
        ]

        # We need to re-zip gradients and variables as grads_and_vars can be only unzipped once.
        grads_and_vars = zip(gradients, variables)

        # We can overlap model averaging and local SGD [2].
        with tf.control_dependencies(assign_ops):
            return apply_grads_func(grads_and_vars, **kwargs)

    def apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        g, v = list(zip(*grads_and_vars))

        return tf.cond(self._cond_var_Ada_var,
                       lambda: self._sma(apply_grads_func, g, v, **kwargs),
                       lambda: self._ssgd(apply_grads_func, g, v, **kwargs))
