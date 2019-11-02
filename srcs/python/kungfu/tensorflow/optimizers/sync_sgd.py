import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.tensorflow.ops import (current_cluster_size, defuse, fuse,
                                   group_all_reduce, group_nccl_all_reduce)

from .core import _create_kungfu_optimizer, _KungFuAlgorithm


def SynchronousSGDOptimizer(optimizer,
                            nccl=False,
                            nccl_fusion=True,
                            name=None,
                            use_locking=False):
    """SynchronousSGDOptimizer implements the [S-SGD]_ algorithm.

    This optimizer is equivalent to the DistributedOptimizer in Horovod.
    Every iteration of training, this optimizer computes the averaged gradients
    to correct diverged model replicas.

    .. [S-SGD] Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, 2017, `S-SGD Paper <https://arxiv.org/pdf/1706.02677>`_

    Arguments:
        optimizer {tf.train.Optimizer, tf.keras.optimizers.Optimizer} -- Optimizer to use for computing gradients and applying updates.

    Keyword Arguments:
        - nccl {bool} -- using NCCL to average gradients. (default: {False})
        - nccl_fusion {bool} -- fusing all gradients to amortise NCCL operation launch cost. (default: {True})
        - name {str} -- name prefix for the operations created when applying gradients. Defaults to "KungFu" followed by the provided optimizer type. (default: {None})
        - use_locking {bool} -- Whether to use locking when updating variables. (default: {False})

    Raises:
        TypeError: Wrapped optimizer is not a subclass of tf.train.Optimizer or tf.keras.optimizers.Optimizer

    Returns:
        optimizer {tf.train.Optimizer, tf.keras.optimizers.Optimizer} -- KungFu distributed optimizer
    """
    sync_sgd_algo = _SynchronousSGD(nccl, nccl_fusion)
    return _create_kungfu_optimizer(optimizer, sync_sgd_algo, name,
                                    use_locking)


class _SynchronousSGD(_KungFuAlgorithm):
    def __init__(self, nccl=False, nccl_fusion=True):
        self._nccl = nccl
        self._nccl_fusion = nccl_fusion
        self._num_workers = current_cluster_size()

    def apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        gradients, variables = list(zip(*grads_and_vars))

        if self._nccl:
            # FIXME: We have a limitation that KungFu schedules NCCL operations
            # in the order of the given gradients. This order is sub-optimal
            # to the topological sorting order of dataflow. We get around of this issue by
            # fusing all gradients. We need to figure out H ow to get the optimal topological s
            # sortting order from TensorFlow.
            if self._nccl_fusion:
                fused_grad = fuse(gradients)
                summed_fused_gradients = group_nccl_all_reduce([fused_grad])
                summed_gradients = defuse(summed_fused_gradients[0],
                                          [g.shape for g in gradients])
            else:
                summed_gradients = group_nccl_all_reduce(gradients)
        else:
            summed_gradients = group_all_reduce(gradients)

        reduced_grads = map_maybe(lambda g: g / self._num_workers,
                                  summed_gradients)

        # We need to re-zip gradients and variables as grads_and_vars can be only unzipped once.
        reduced_grads_and_vars = zip(reduced_grads, variables)

        return apply_grads_func(reduced_grads_and_vars, **kwargs)
