import keras
from kungfu.tensorflow.compat import _tf_assign, _tf_major_version
from kungfu.tensorflow.ops import broadcast
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp


class KerasBroadcastGlobalVariablesCallback(keras.callbacks.Callback):
    """A Keras callback that broadcasts global variables at the begining of training."""
    def __init__(self, *args):
        super(KerasBroadcastGlobalVariablesCallback, self).__init__(*args)
        self.broadcast_done = False

    def on_batch_end(self, batch, logs=None):
        """broadcast should be done after the first gradient step to ensure optimizer initialization."""
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
            keras.backend.get_session().run(BroadcastGlobalVariablesOp())

        self.broadcast_done = True
