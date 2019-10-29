import tensorflow as tf


class KerasInitCallback(tf.keras.callbacks.Callback):
    """Initialize the KungFu step variable using session.

    This callback is given as a hook to the keras.model.compile() method.

    """
    def on_train_begin(self, logs=None):
        tf.keras.backend.get_session().run(tf.global_variables_initializer())
