import tensorflow as tf
from kungfu.ops import (broadcast, current_cluster_size, current_rank,
                        group_all_reduce)

from .core import KungFuOptimizer


class SyncModelAveragingSGDOptimizer(KungFuOptimizer):
    """SyncModelAveragingSGDOptimizer implements a simple version of Synchronous Model Averaging (SMA).

    [1] CROSSBOW: Scaling Deep Learning with Small Batch Sizes on Multi-GPU Servers, VLDB 2019
    http://www.vldb.org/pvldb/vol12/p1399-koliousis.pdf

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "Distributed" followed by the provided
        optimizer type.
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.

    """
    def __init__(self, optimizer, name=None, use_locking=False):
        super(SyncModelAveragingSGDOptimizer,
              self).__init__(optimizer, name, use_locking=use_locking)
        self._num_workers = current_cluster_size()
        self._rank = current_rank()

    def apply_gradients(self, grads_and_vars, **kwargs):
        # Compute and apply an averaged model
        _, variables = list(zip(*grads_and_vars))
        sum_vars = group_all_reduce(variables)
        avg_vars = [g / self._num_workers for g in sum_vars]
        assign_ops = [
            tf.assign(v, avg_v) for v, avg_v in zip(variables, avg_vars)
        ]

        # We overlap model averaging and local SGD.
        # This overlapping is proposed in [1]
        with tf.control_dependencies(assign_ops):
            return self._optimizer.apply_gradients(grads_and_vars, **kwargs)

    def distributed_initializer(self):
        ops = [tf.assign(v, broadcast(v)) for v in self.variables()]
        return tf.group(ops)