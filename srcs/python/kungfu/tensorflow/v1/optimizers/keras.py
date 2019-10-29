import tensorflow as tf


class KerasInitCallback(tf.keras.callbacks.Callback):
    """Initialize variables on distributed workers.

    This callback is given as a hook to the keras.model.compile() method.

    """
    def on_train_begin(self, logs=None):
        if hasattr(self.model.optimizer.optimizer, 'distributed_initializer'):
            tf.keras.backend.get_session().run(
                self.model.optimizer.optimizer.distributed_initializer())
