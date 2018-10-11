import tensorflow as tf

from .negotiator import NegotiableOptimizer

__all__ = [
    'AsyncSGDOptimizer',
]


def _load_op_lib(load_from_extension):
    if load_from_extension:
        name = 'kungfu_tensorflow_lib.cpython-36m-darwin.so'  # FIXME: auto determin the name
        return tf.load_op_library(name)
    else:
        name = 'negotiator'
        import platform
        suffix = 'so' if platform.uname()[0] != 'Darwin' else 'dylib'
        return tf.load_op_library('./lib/lib%s.%s' % (name, suffix))


_op_lib = _load_op_lib(False)

negotiator = _op_lib.negotiator

# TODO: tf.ops.NotDifferentiable('negotiator')


class AsyncSGDOptimizer(NegotiableOptimizer):
    """An optimizer that negotiates using the negotiator operator."""

    def _negotiate_grad(self, grad):
        """Negotiate grad with peers."""
        with tf.variable_scope('NegotiatedGrad'):
            return negotiator(grad)
