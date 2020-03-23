import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.tensorflow.compat import _tf_assign, _tf_hook
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import (counter, current_cluster_size, current_rank,
                                   group_all_reduce, fuse, defuse, request_variable, 
                                   request_variable_with_template, save_variable, barrier)
from kungfu.tensorflow.optimizers.core import (_create_kungfu_keras_optimizer,
                                               _create_kungfu_optimizer,
                                               _KungFuAlgorithm)
from kungfu.tensorflow.optimizers.async_sgd import (get_random_peer)


def CustomAdaSGDOptimizer(optimizer,
                         alpha=0.1,
                         fuse_requests=True,
                         fused_model_name=None,
                         name=None,
                         use_locking=False,
                         with_keras=False):

    algo = _CustomAdaSGD(alpha, fuse_requests, fused_model_name)
    if not with_keras:
        return _create_kungfu_optimizer(optimizer, algo, name, use_locking)
    else:
        return _create_kungfu_keras_optimizer(optimizer, algo)


class _CustomAdaSGD(_KungFuAlgorithm):
    def __init__(self, alpha, fuse_requests, fused_model_name):
        self._num_workers = current_cluster_size()
        self._alpha = alpha
        self._global_step = tf.train.get_or_create_global_step()
        self._cond_var_Ada_var = tf.Variable(False, trainable=False, name='cond_var_Ada')
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
    
    def _async_sgd(self, apply_grads_func, gradients, variables, **kwargs):
        np, rank = current_cluster_size(), current_rank()
        target = get_random_peer(np, rank)

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

    def apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        g, v = list(zip(*grads_and_vars))

        return tf.cond(self._cond_var_Ada_var,
                       lambda: self._sma(apply_grads_func, g, v, **kwargs),
                       lambda: self._ssgd(apply_grads_func, g, v, **kwargs))
