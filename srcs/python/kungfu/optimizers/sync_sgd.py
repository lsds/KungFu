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

    def compute_gradients(self, *args, **kwargs):
        grads_and_vars = super(SyncSGDOptimizer, self).compute_gradients(*args, **kwargs)
        gradients, variables = list(zip(*grads_and_vars))
        summed_gradients = group_all_reduce(gradients)
        if self._average:
            reduced_grads = [g / self._num_workers for g in summed_gradients]
        else:
            reduced_grads = summed_gradients
        return list(zip(reduced_grads, variables))

    def distributed_initializer(self):
        ops = [tf.assign(v, broadcast(v)) for v in self.model_variables()]
        return tf.group(ops)
