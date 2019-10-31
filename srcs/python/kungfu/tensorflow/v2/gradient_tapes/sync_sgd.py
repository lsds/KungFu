import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.tensorflow.v1.ops import (current_cluster_size, group_all_reduce)


def SynchronousSGDGradientTape(gradtape):
    """A tape that wraps another tf.GradientTape, using an allreduce to
    average gradient values before applying gradients to model weights.
    Args:
        gradtape:
        GradientTape to use for computing gradients and applying updates.
    """
    cls = type(gradtape.__class__.__name__, (gradtape.__class__, ),
               dict(_SynchronousSGDGradientTape.__dict__))
    if hasattr(gradtape, '_watch_accessed_variables'):
        return cls(gradtape._tape, gradtape._persistent,
                   gradtape._watch_accessed_variables)
    else:
        return cls(gradtape._tape, gradtape._persistent)


class _SynchronousSGDGradientTape(tf.GradientTape):
    """SynchronousSGDGradientTape implements the [S-SGD]_ algorithm.

    .. [S-SGD] Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, 2017, `S-SGD Paper <https://arxiv.org/pdf/1706.02677>`_

    Args:
      tape:
        Tape to use for computing gradients and applying updates.

    """
    def __init__(self, tape, persistent=False, watch_accessed_variables=True):
        if hasattr(tape, '_watch_accessed_variables'):
            super(self.__class__, self).__init__(persistent,
                                                 watch_accessed_variables)
        else:
            super(self.__class__, self).__init__(persistent)
        self._tape = tape
        self._num_workers = current_cluster_size()

    def gradient(self, target, sources, output_gradients=None):
        gradients = super(self.__class__,
                          self).gradient(target, sources, output_gradients)
        summed_gradients = group_all_reduce(gradients)
        reduced_grads = map_maybe(lambda g: g / self._num_workers,
                                  summed_gradients)
        return reduced_grads
