import tensorflow as tf
from kungfu.tensorflow.v1.ops import broadcast

__all__ = [
    'BroadcastGlobalVariablesHook',
    'BroadcastGlobalVariablesOp',
    'BroadcastGlobalVariablesCallback',
]


def BroadcastGlobalVariablesOp():
    ops = [tf.assign(v, broadcast(v)) for v in tf.global_variables()]
    return tf.group(ops)


class BroadcastGlobalVariablesHook(tf.estimator.SessionRunHook):
    def __init__(self):
        super(BroadcastGlobalVariableHook, self).__init__()
        self.bcast_op = None

    def begin(self):
        self.bcast_op = BroadcastGlobalVariablesOp()

    def after_create_session(self, session, coord):
        session.run(self.bcast_op)


class BroadcastGlobalVariablesCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        tf.keras.backend.get_session().run(BroadcastGlobalVariablesOp())
