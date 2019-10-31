import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.tensorflow.v1.ops import (broadcast, current_cluster_size,
                                      current_rank, group_all_reduce)


class SynchronousSGDOptimizer(tf.keras.optimizers.Optimizer):
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
                 name=None,
                 use_locking=False):
        super(SynchronousSGDOptimizer, self).__init__(name=name)
        self._optimizer = optimizer
        self._num_workers = current_cluster_size()
        self._rank = current_rank()

    def apply_gradients(self, grads_and_vars, **kwargs):
        gradients, variables = list(zip(*grads_and_vars))
        
        # for var in variables:
        #     var.assign(broadcast(var))
        
        summed_gradients = group_all_reduce(gradients)

        reduced_grads = map_maybe(lambda g: g / self._num_workers,
                                  summed_gradients)
        reduced_grads_and_vars = zip(reduced_grads, variables)
        return self._optimizer.apply_gradients(reduced_grads_and_vars,
                                               **kwargs)

    def get_config(self):
        return self._optimizer.optimizer.get_config()
