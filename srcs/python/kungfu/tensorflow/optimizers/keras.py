import keras


class KungFuKerasOptimizer(keras.optimizers.Optimizer):
    def __init__(self, optimizer, algo):
        super(KungFuKerasOptimizer, self).__init__()
        self._optimizer = optimizer
        self._algo = algo

    def get_gradients(self, loss, params):
        return self._optimizer.get_gradients(loss, params)

    def apply_gradients(self, grads_and_vars, **kwargs):
        return self._algo.apply_gradients(self._optimizer.apply_gradients,
                                          grads_and_vars, **kwargs)

    def get_updates(self, loss, params):
        return self._optimizer.get_updates(loss, params)

    def get_config(self):
        return self._optimizer.get_config()
