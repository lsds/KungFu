import tensorflow as tf
from kungfu.internal import _get_num_peers
from kungfu.ops import all_reduce, global_variance, group_all_reduce

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

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        """Negotiate grads with peers, using plain allreduce."""
        gradients, variables = list(zip(*grads_and_vars_to_negotiate))
        summed_gradients = group_all_reduce(gradients)
        if self._average:
            reduced_grads = [g / self._num_workers for g in summed_gradients]
        else:
            reduced_grads = summed_gradients
        return list(zip(reduced_grads, variables))


class MonSyncSGDOptimizer(KungFuOptimizer):
    """An optimizer that reduce gradients for synchronisation and compute the varience of gradients for monitoring."""
    def __init__(self, optimizer, name=None, use_locking=False):
        super(MonSyncSGDOptimizer, self).__init__(optimizer, name, use_locking)

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        """Negotiate grad with peers, following flexible strategy."""

        negotiated_grad_and_vars = []
        for grad, var in grads_and_vars_to_negotiate:
            with tf.variable_scope('NegotiatedGrad'):
                with tf.control_dependencies([global_variance(grad)]):
                    negotiated_grad_and_vars.append((all_reduce(grad), var))
        return negotiated_grad_and_vars
