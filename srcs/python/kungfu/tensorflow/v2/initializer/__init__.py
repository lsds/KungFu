import tensorflow as tf
from kungfu.tensorflow.v1.ops import broadcast
from tensorflow import keras

__all__ = [
    'BroadcastGlobalVariablesCallback',
]


class BroadcastGlobalVariablesCallback(keras.callbacks.Callback):
    def __init__(self, *args):
        super(BroadcastGlobalVariablesCallback, self).__init__(*args)
        self.broadcast_done = False

    def on_batch_end(self, batch, logs=None):
        if self.broadcast_done:
            return

        if hasattr(self.model, 'variables'):
            for v in self.model.variables:
                broadcast(v)

            opt_variables = None
            if hasattr(self.model.optimizer, 'variables'):
                opt_variables = self.model.optimizer.variables()
            else:
                opt_variables = self.model.optimizer.optimizer.variables()
            for v in opt_variables:
                broadcast(v)
        else:
            raise RuntimeError('No variables() in %s', self.model)

        self.broadcast_done = True
