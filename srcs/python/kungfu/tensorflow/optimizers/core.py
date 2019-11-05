import tensorflow as tf
from kungfu.tensorflow.compat import _tf_optimizer
from kungfu.tensorflow.ops import counter


class KungFuTFOptimizer(_tf_optimizer):
    def __init__(self, optimizer, algo, name, use_locking=False):
        super(KungFuTFOptimizer, self).__init__(name=name,
                                                use_locking=use_locking)
        self._optimizer = optimizer
        self._algo = algo

    def apply_gradients(self, *args, **kwargs):
        return self._algo.apply_gradients(self._optimizer.apply_gradients,
                                          *args, **kwargs)

    def compute_gradients(self, *args, **kwargs):
        return self._optimizer.compute_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self._optimizer.variables(*args, **kwargs)


class KungFuTFKerasOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, algo, name):
        super(KungFuTFKerasOptimizer, self).__init__(name=name)
        self._optimizer = optimizer
        self._algo = algo

    def apply_gradients(self, grads_and_vars, **kwargs):
        return self._algo.apply_gradients(self._optimizer.apply_gradients,
                                          grads_and_vars, **kwargs)

    def get_config(self):
        return self._optimizer.optimizer.get_config()


class _KungFuAlgorithm:
    def apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        raise NotImplementedError('Must be implemented by sub-class.')


def _create_kungfu_optimizer(optimizer, kungfu_algo, name, use_locking):
    if name is None:
        name = "KungFu{}".format(type(optimizer).__name__)

    if isinstance(optimizer, _tf_optimizer):
        return KungFuTFOptimizer(optimizer,
                                 kungfu_algo,
                                 name,
                                 use_locking=use_locking)
    elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
        return KungFuTFKerasOptimizer(optimizer, kungfu_algo, name)
    else:
        raise TypeError(
            'Given optimizer must be tf.train.Optimizer or tf.keras.optimizers.Optimizer'
        )


def _create_kungfu_keras_optimizer(optimizer, kungfu_algo):
    import keras
    if isinstance(optimizer, keras.optimizers.Optimizer):
        from .keras import KungFuKerasOptimizer
        return KungFuKerasOptimizer(optimizer, kungfu_algo)
    else:
        raise TypeError('Given optimizer must be keras.optimizers.Optimizer')
