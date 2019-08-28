import tensorflow as tf
from kungfu.ops import all_reduce, global_variance, group_all_reduce

from kungfu.optimizers.core import KungFuOptimizer

from kungfu.ops import _concat, peer_info, global_gradient_noise_scale


class GradientNoiseScaleAdaptiveOptimizer(KungFuOptimizer):
    def __init__(self,
                 optimizer,
                 local_batch_size,
                 name=None,
                 use_locking=False):
        super(GradientNoiseScaleAdaptiveOptimizer,
              self).__init__(optimizer, name, use_locking)
        self._local_batch_size = local_batch_size

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        gradients, variables = list(zip(*grads_and_vars_to_negotiate))
        reduced_gradients = group_all_reduce(gradients)

        for g in gradients:
            print('%s :: %s' % (g.name, g.shape))

        concat_grad = _concat(gradients)
        concat_negotiated_grad = _concat(reduced_gradients)
        gns = global_gradient_noise_scale(self._local_batch_size, concat_grad,
                                          concat_negotiated_grad)
        self._gns = gns
        return list(zip(reduced_gradients, variables))

    # def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
    #     """Negotiate grad with peers, following flexible strategy."""

    #     negotiated_grad_and_vars = []
    #     for grad, var in grads_and_vars_to_negotiate:
    #         with tf.variable_scope('NegotiatedGrad'):
    #             with tf.control_dependencies([global_variance(grad)]):
    #                 negotiated_grad_and_vars.append((all_reduce(grad), var))
    #     return negotiated_grad_and_vars
