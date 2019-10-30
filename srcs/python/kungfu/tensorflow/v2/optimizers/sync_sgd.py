import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.tensorflow.v1.ops import (broadcast, current_cluster_size,
                                      group_all_reduce)


class SynchronousSGDOptimizer(tf.compat.v1.train.Optimizer):
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
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.

    """
    def __init__(self, optimizer, name=None, use_locking=False):
        if name is None:
            name = "KungFu{}".format(type(optimizer).__name__)
        super(SynchronousSGDOptimizer, self).__init__(name=name,
                                                      use_locking=use_locking)
        self._num_workers = current_cluster_size()
        self._optimizer = optimizer

    def apply_gradients(self, grads_and_vars, **kwargs):
        gradients, variables = list(zip(*grads_and_vars))
        summed_gradients = group_all_reduce(gradients)
        reduced_grads = map_maybe(lambda g: g / self._num_workers,
                                  summed_gradients)
        reduced_grads_and_vars = zip(reduced_grads, variables)
        return self._optimizer.apply_gradients(reduced_grads_and_vars,
                                               **kwargs)

    def compute_gradients(self, *args, **kwargs):
        return self._optimizer.compute_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self._optimizer.variables(*args, **kwargs)
