import tensorflow as tf


class KerasInitCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        tf.keras.backend.get_session().run(tf.global_variables_initializer())
