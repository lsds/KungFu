import tensorflow as tf
from kungfu.ops import all_reduce, global_variance, group_all_reduce, all_reduce

from kungfu.optimizers.core import KungFuOptimizer

from kungfu.ops import _concat, peer_info, global_gradient_noise_scale


def predict_batch_size(gns):
    # return tf.sqrt(0.96 * tf.maximum(tf.constant(0, dtype=tf.float32), gns))
    return tf.sqrt(0.96 * tf.abs(gns))


class GradientNoiseScaleAdaptiveOptimizer(KungFuOptimizer):
    def __init__(self,
                 optimizer,
                 local_batch_size,
                 decay=0.01,
                 name=None,
                 use_locking=False):
        super(GradientNoiseScaleAdaptiveOptimizer,
              self).__init__(optimizer, name, use_locking)
        self._local_batch_size = local_batch_size
        self._decay = decay
        self._rank, self._np = peer_info(tf.constant(-1, dtype=tf.int32))

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        gradients, variables = list(zip(*grads_and_vars_to_negotiate))
        summed_gradients = group_all_reduce(gradients)
        reduced_gradients = [g / tf.cast(self._np, dtype=tf.float32) for g in summed_gradients]

        for g in gradients:
            print('%s :: %s' % (g.name, g.shape))

        concat_grad = _concat(gradients)
        concat_negotiated_grad = _concat(summed_gradients)
        gns = global_gradient_noise_scale(self._local_batch_size, concat_grad,
                                          concat_negotiated_grad, self._decay)

        self._gns = gns
        bs = predict_batch_size(gns)
        global_bs = all_reduce(bs)
        self._predicated_local_batch_size = bs
        self._predicated_global_batch_size = global_bs

        return list(zip(reduced_gradients, variables))

    # def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
    #     """Negotiate grad with peers, following flexible strategy."""

    #     negotiated_grad_and_vars = []
    #     for grad, var in grads_and_vars_to_negotiate:
    #         with tf.variable_scope('NegotiatedGrad'):
    #             with tf.control_dependencies([global_variance(grad)]):
    #                 negotiated_grad_and_vars.append((all_reduce(grad), var))
    #     return negotiated_grad_and_vars
