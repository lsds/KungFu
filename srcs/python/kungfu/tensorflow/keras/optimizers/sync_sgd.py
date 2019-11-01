import tensorflow as tf
from kungfu.tensorflow.optimizers.sync_sgd import _SynchronousSGD
from .core import KungFuKerasOptimizer


class SynchronousSGDOptimizer(KungFuKerasOptimizer):
    """SynchronousSGDOptimizer implements the [S-SGD]_ algorithm.

    This optimizer is equivalent to the DistributedOptimizer in Horovod.
    Every iteration of training, this optimizer computes the averaged gradients
    to correct diverged model replicas.

    .. [S-SGD] Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, 2017, `S-SGD Paper <https://arxiv.org/pdf/1706.02677>`_

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "KungFu" followed by the provided
        optimizer type.
    """
    def __init__(self,
                 optimizer,
                 nccl=False,
                 nccl_fusion=True,
                 name=None):
        algo = _SynchronousSGD(nccl, nccl_fusion)
        super(SynchronousSGDOptimizer, self).__init__(optimizer,
                                                      algo,
                                                      name)
