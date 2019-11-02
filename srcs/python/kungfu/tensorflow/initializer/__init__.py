import tensorflow as tf
from kungfu.tensorflow.compat import _tf_assign, _tf_hook, _tf_major_version
from kungfu.tensorflow.ops import broadcast
from tensorflow import keras

__all__ = [
    'BroadcastGlobalVariablesHook', 'BroadcastGlobalVariablesOp',
    'BroadcastGlobalVariablesCallback', 'broadcast_variables'
]


def broadcast_variables(variables):
    for v in variables:
        _tf_assign(v, broadcast(v))


def BroadcastGlobalVariablesOp():
    ops = [tf.assign(v, broadcast(v)) for v in tf.global_variables()]
    return tf.group(ops)


class BroadcastGlobalVariablesHook(_tf_hook):
    def __init__(self):
        super(BroadcastGlobalVariablesHook, self).__init__()
        self.bcast_op = None

    def begin(self):
        self.bcast_op = BroadcastGlobalVariablesOp()

    def after_create_session(self, session, coord):
        session.run(self.bcast_op)


class BroadcastGlobalVariablesCallback(tf.keras.callbacks.Callback):
    def __init__(self, *args):
        super(BroadcastGlobalVariablesCallback, self).__init__(*args)
        self.broadcast_done = False

    def on_batch_end(self, batch, logs=None):
        if self.broadcast_done:
            return

        if _tf_major_version == 2:
            if hasattr(self.model, 'variables'):
                for v in self.model.variables:
                    _tf_assign(v, broadcast(v))

                opt_variables = None
                if hasattr(self.model.optimizer, 'variables'):
                    opt_variables = self.model.optimizer.variables()
                else:
                    opt_variables = self.model.optimizer.optimizer.variables()

                # print(opt_variables)
                for v in opt_variables:
                    _tf_assign(v, broadcast(v))
            else:
                raise RuntimeError('No variables() in %s', self.model)

        if _tf_major_version == 1:
            tf.keras.backend.get_session().run(BroadcastGlobalVariablesOp())

        self.broadcast_done = True
