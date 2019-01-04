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


def lazy_load_op_lib():
    global _op_lib
    if _op_lib is None:
        _op_lib = _load_op_lib(_op_lib_name)
    return _op_lib


class AsyncSGDOptimizer(NegotiableOptimizer):
    """An optimizer that negotiates using the negotiator operator."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse='',
                 use_global_step=True):
        super(AsyncSGDOptimizer, self).__init__(optimizer, name, use_locking,
                                                device_dense, device_sparse)

        self._op_lib = lazy_load_op_lib()

        self._use_global_step = use_global_step
        if self._use_global_step:
            self._trained_steps = tf.Variable(tf.zeros([], tf.int32))
            self._modify_trained_steps = tf.assign(
                self._trained_steps,
                self._op_lib.global_step_modifier(self._trained_steps))

    def _negotiate_grad(self, grad):
        """Negotiate grad with peers."""

        def build_op():
            with tf.variable_scope('NegotiatedGrad'):
                return self._op_lib.negotiator(grad)

        if self._use_global_step:
            with tf.control_dependencies([self._modify_trained_steps]):
                return build_op()
        else:
            return build_op()

    def _set_gradient_count(self, n):
        return self._op_lib.set_gradient_count(tf.constant(n, tf.int32))
