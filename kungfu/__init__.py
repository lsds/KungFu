import tensorflow as tf

from .negotiator import NegotiableOptimizer

__all__ = [
    'AsyncSGDOptimizer',
]


def _load_op_lib():
    name = 'kungfu_tensorflow_ops'
    import sysconfig
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    from tensorflow.python.platform import resource_loader  # FIXME: don't depend on TF
    filename = resource_loader.get_path_to_datafile(name + suffix)
    return tf.load_op_library(filename)


_op_lib = _load_op_lib()

negotiator = _op_lib.negotiator

# TODO: tf.ops.NotDifferentiable('negotiator')


class AsyncSGDOptimizer(NegotiableOptimizer):
    """An optimizer that negotiates using the negotiator operator."""

    def _negotiate_grad(self, grad):
        """Negotiate grad with peers."""
        with tf.variable_scope('NegotiatedGrad'):
            return negotiator(grad)
