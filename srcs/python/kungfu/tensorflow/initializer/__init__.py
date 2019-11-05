import tensorflow as tf
from kungfu.tensorflow.compat import _tf_assign, _tf_hook, _tf_major_version
from kungfu.tensorflow.ops import broadcast

__all__ = [
    'broadcast_variables',
    'BroadcastGlobalVariablesCallback',
    'BroadcastGlobalVariablesHook',
    'BroadcastGlobalVariablesOp',
]


def broadcast_variables(variables):
    """A TensorFlow function that broadcasts global variables.

    This function is often used with ``tf.GradientTape`` or embedded as part of a training program.
    """
    for v in variables:
        _tf_assign(v, broadcast(v))


def BroadcastGlobalVariablesOp():
    """A TensorFlow operator that broadcasts global variables.

    This operator if often used with the low-level tf.Session
    """
    ops = [tf.assign(v, broadcast(v)) for v in tf.global_variables()]
    return tf.group(ops)


class BroadcastGlobalVariablesHook(_tf_hook):
    """A TensorFlow hook that broadcasts global variables at the begining of training.

    This hook is often used with ``tf.session.MonitoredSession`` and ``tf.train.Estimator``.
    """
    def __init__(self):
        super(BroadcastGlobalVariablesHook, self).__init__()
        self.bcast_op = None

    def begin(self):
        """Create a broadcast op at the beginning."""
        self.bcast_op = BroadcastGlobalVariablesOp()

    def after_create_session(self, session, coord):
        """Broadcast global vartiables after creating the session."""
        session.run(self.bcast_op)


def BroadcastGlobalVariablesCallback(with_keras=False):
    """Keras callback that broadcasts global variables at the begining of training.

    Keyword Arguments:
        with_keras {bool} -- Runs with pure Keras or not (default: {False})

    Returns:
        {tf.keras.callbacks.Callback, keras.callbacks.Callback} -- Callback
    """
    if not with_keras:
        return _TFKerasBroadcastGlobalVariablesCallback()
    else:
        from .keras import _KerasBroadcastGlobalVariablesCallback
        return _KerasBroadcastGlobalVariablesCallback()


def _keras_callback_on_batch_end(callback, batch, logs=None):
    """broadcast should be done after the first gradient step to ensure optimizer initialization."""
    if callback.broadcast_done:
        return

    if _tf_major_version == 2:
        if hasattr(callback.model, 'variables'):
            for v in callback.model.variables:
                _tf_assign(v, broadcast(v))

            opt_variables = None
            if hasattr(callback.model.optimizer, 'variables'):
                opt_variables = callback.model.optimizer.variables()
            else:
                opt_variables = callback.model.optimizer.optimizer.variables()

            # print(opt_variables)
            for v in opt_variables:
                _tf_assign(v, broadcast(v))
        else:
            raise RuntimeError('No variables() in %s', callback.model)

    if _tf_major_version == 1:
        tf.keras.backend.get_session().run(BroadcastGlobalVariablesOp())

    callback.broadcast_done = True


class _TFKerasBroadcastGlobalVariablesCallback(tf.keras.callbacks.Callback):
    def __init__(self, *args):
        super(_TFKerasBroadcastGlobalVariablesCallback, self).__init__(*args)
        self.broadcast_done = False

    def on_batch_end(self, batch, logs=None):
        return _keras_callback_on_batch_end(self, batch, logs)
