import tensorflow as tf
from kungfu.internal import _get_num_peers
from kungfu.ops import all_reduce, broadcast, global_variance, group_all_reduce

from .core import KungFuOptimizer


class SyncSGDOptimizer(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""
    def __init__(self,
                 optimizer,
                 average_gradients=True,
                 name=None,
                 use_locking=False):
        super(SyncSGDOptimizer, self).__init__(optimizer, name, use_locking)
        self._average = average_gradients
        self._num_workers = _get_num_peers()  # FIXME: use a variable

    def apply_gradients(self, grads_and_vars, **kwargs):
        gradients, variables = list(zip(*grads_and_vars))
        summed_gradients = group_all_reduce(gradients)
        if self._average:
            reduced_grads = [g / self._num_workers for g in summed_gradients]
        else:
            reduced_grads = summed_gradients
        reduced_grads_and_vars = zip(reduced_grads, variables)
        return self._optimizer.apply_gradients(reduced_grads_and_vars,
                                               **kwargs)

    def distributed_initializer(self):
        ops = [tf.assign(v, broadcast(v)) for v in self.model_variables()]
        return tf.group(ops)


class SyncSGDWithGradVarianceOptimizer(KungFuOptimizer):
    def __init__(self,
                 optimizer,
                 name=None,
                 monitor_interval=1,
                 use_locking=False):
        super(SyncSGDWithGradVarianceOptimizer,
              self).__init__(optimizer, name, use_locking)
        self._num_workers = _get_num_peers()  # FIXME: use a variable
        self._interval = monitor_interval
        self._step = tf.Variable(0, trainable=False, dtype=tf.int32)

    def _monitor(self, grads):
        square_grads = [tf.square(g) for g in grads]
        summed_square_grads = group_all_reduce(square_grads)
        reduced_square_grads = [
            g / self._num_workers for g in summed_square_grads
        ]
        grad_variances = [
            square_grad - tf.square(grad)
            for square_grad, grad in zip(reduced_square_grads, reduced_grads)
        ]
        self._variances = [
            tf.norm(grad_variance) for grad_variance in grad_variances
        ]
        self._summed_variance = tf.reduce_sum(self._variances)
        print_op = tf.print('summed variance:', self._summed_variance)

        with tf.control_dependencies([print_op]):
            return tf.identity(self._summed_variance)

    def apply_gradients(self, grads_and_vars, **kwargs):
        grads, variables = list(zip(*grads_and_vars))
        summed_grads = group_all_reduce(grads)
        reduced_grads = [g / self._num_workers for g in summed_grads]

        step_add_op = tf.assign_add(self._step, 1)
        monitor_op = tf.cond(
            tf.mod(self._step, self._interval) == 0, _monitor(grads),
            lambda: tf.no_op)
        with tf.control_dependencies([step_add_op]):
            with tf.control_dependencies([monitor_op]):
                return self._optimizer.apply_gradients(
                    zip(reduced_grads, variables), **kwargs)

    def distributed_initializer(self):
        ops = [tf.assign(v, broadcast(v)) for v in self.model_variables()]
        return tf.group(ops)
