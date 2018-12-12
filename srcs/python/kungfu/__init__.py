import os
import sysconfig

import tensorflow as tf

from .negotiator import NegotiableOptimizer

__all__ = [
    'AsyncSGDOptimizer',
]


def _load_op_lib(name):
    module_path = os.path.dirname(__file__)
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    filename = os.path.join(module_path, name + suffix)
    return tf.load_op_library(filename)


_op_lib_name = 'kungfu_tensorflow_ops'
_op_lib = None


class AsyncSGDOptimizer(NegotiableOptimizer):
    """An optimizer that negotiates using the negotiator operator."""

    def _negotiate_grad(self, grad):
        """Negotiate grad with peers."""

        with tf.variable_scope('NegotiatedGrad'):
            global _op_lib
            if _op_lib is None:
                _op_lib = _load_op_lib(_op_lib_name)
            negotiator = _op_lib.negotiator
            # TODO: tf.ops.NotDifferentiable('negotiator')
            return negotiator(grad)
