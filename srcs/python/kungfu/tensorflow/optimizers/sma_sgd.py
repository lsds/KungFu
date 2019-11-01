import tensorflow as tf
from kungfu.tensorflow import _tf_assign
from kungfu.tensorflow.v1.ops import (current_cluster_size,
                                      current_rank, group_all_reduce)

from .core import KungFuTFOptimizer


class SynchronousAveragingOptimizer(KungFuTFOptimizer):
    """SynchronousAveragingOptimizer implements the [SMA]_ algorithm.

    [EA-SGD]_ proposed to use model averaging to train deep learning models and prove its convergence.
    [SMA]_ further improves [EA-SGD]_ results and show model averaging can benefit small-batch training
    and achieves fast convergence compared to synchronous SGD.

    .. [EA-SGD] Deep learning with Elastic Averaging SGD, NIPS 2015, `EA-SGD Paper <https://arxiv.org/abs/1412.6651>`_
    .. [SMA] CrossBow: Scaling Deep Learning with Small Batch Sizes on Multi-GPU Servers, VLDB 2019, `SMA Paper <http://www.vldb.org/pvldb/vol12/p1399-koliousis.pdf>`_

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "KungFu" followed by the provided
        optimizer type.
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.

    """
    def __init__(self, optimizer, name=None, use_locking=False):
        algo = _SynchronousAveraging()
        super(SynchronousAveragingOptimizer, self).__init__(optimizer,
                                                      algo,
                                                      name,
                                                      use_locking=use_locking)

class _SynchronousAveraging:
    def __init__(self):
        self._num_workers = current_cluster_size()
        self._rank = current_rank()

    def apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        # It is important to apply model averaging every iteration [2]
        gradients, variables = list(zip(*grads_and_vars))
        sum_vars = group_all_reduce(variables)
        avg_vars = [g / self._num_workers for g in sum_vars]

        # TODO: Apply momentum to the averaged model [2]
        assign_ops = [
            _tf_assign(v, avg_v) for v, avg_v in zip(variables, avg_vars)
        ]

        # We need to re-zip gradients and variables as grads_and_vars can be only unzipped once.
        new_grads_and_vars = zip(gradients, variables)

        # We can overlap model averaging and local SGD [2].
        with tf.control_dependencies(assign_ops):
            return apply_grads_func(new_grads_and_vars, **kwargs)
