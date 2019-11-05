import keras
from kungfu.tensorflow.initializer import _keras_callback_on_batch_end


class _KerasBroadcastGlobalVariablesCallback(keras.callbacks.Callback):
    """A Keras callback that broadcasts global variables at the begining of training."""
    def __init__(self, *args):
        super(_KerasBroadcastGlobalVariablesCallback, self).__init__(*args)
        self.broadcast_done = False

    def on_batch_end(self, batch, logs=None):
        return _keras_callback_on_batch_end(self, batch, logs)
