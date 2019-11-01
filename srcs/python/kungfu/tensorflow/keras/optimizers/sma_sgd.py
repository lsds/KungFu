import tensorflow as tf
from kungfu.tensorflow.optimizers.sma_sgd import _SynchronousAveraging
from .core import KungFuKerasOptimizer


class SynchronousAveragingOptimizer(KungFuKerasOptimizer):
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

    """
    def __init__(self,
                 optimizer,
                 nccl=False,
                 nccl_fusion=True,
                 name=None):
        algo = _SynchronousAveraging()
        super(SynchronousAveragingOptimizer, self).__init__(optimizer,
                                                      algo,
                                                      name)
