import tensorflow as tf
from kungfu.ops import group_all_reduce

from .core import KungFuOptimizer


class ParallelOptimizer(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse='',
                 use_global_step=True,
                 device_batch_size=None):
        super(ParallelOptimizer, self).__init__(optimizer, name, use_locking,
                                                device_dense, device_sparse)
        self.device_batch_size = device_batch_size

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        """Negotiate grads with peers, using plain allreduce."""
        gradients, variables = list(zip(*grads_and_vars_to_negotiate))
        return list(zip(group_all_reduce(gradients), variables))
