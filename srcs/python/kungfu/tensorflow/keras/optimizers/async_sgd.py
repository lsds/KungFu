import tensorflow as tf
from kungfu.tensorflow.optimizers.async_sgd import _PairAveraging
from .core import KungFuKerasOptimizer

class PairAveragingOptimizer(KungFuKerasOptimizer):
    """PairAveragingOptimizer implements the [AD-PSGD]_ algorithm.

    Every iteration of training, this optimizer:

    1. Randomly selects a peer in the current cluster.
    2. Pulls the selected peer's model
    3. Performs model averaging with the local model.
    4. Applies local gradients
    5. Saves the model to a local store which allows other peers to pull from.

    .. [AD-PSGD] Asynchronous Decentralized Parallel Stochastic Gradient Descent, ICML 2018, `AD-PSGD Paper <https://arxiv.org/abs/1710.06952>`_

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      fuse_requests:
        Fusing the requests for remote variables to amortise communication cost.
        The fusing however takes extra memory and prevents overlapping
        synchronization and training.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "KungFu" followed by the provided
        optimizer type.

    """
    def __init__(self,
                 optimizer,
                 fuse_requests=True,
                 name=None):
        opt_type_name = type(optimizer).__name__
        algo = _PairAveraging(fuse_requests, fused_model_name=opt_type_name)
        super(PairAveragingOptimizer, self).__init__(optimizer,
                                                      algo,
                                                      name)
