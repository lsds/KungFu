import tensorflow as tf
from kungfu.tensorflow.ops import _resize_cluster, counter, step_based_schedule

from .core import _create_kungfu_optimizer
from .sync_sgd import _SynchronousSGD


class _ElasticSynchronousSGD(_SynchronousSGD):
    def __init__(self, config, init_step, *args, **kwargs):
        super(_ElasticSynchronousSGD, self).__init__(*args, **kwargs)
        self._config = config
        self._init_step = init_step

    def _build_resize_op(self, config, init_step):
        step = counter(init_step)
        new_size = step_based_schedule(config, step)
        ckpt_tensor = tf.as_string(step + 1)
        resize_op = _resize_cluster(ckpt_tensor, new_size)
        return resize_op

    def apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        op = super(_ElasticSynchronousSGD,
                   self).apply_gradients(apply_grads_func, grads_and_vars,
                                         **kwargs)
        with tf.control_dependencies([op]):
            return self._build_resize_op(self._config, self._init_step)


def ElasticSyncSGDOptimizer(optimizer,
                            config,
                            init_step,
                            nccl=False,
                            nccl_fusion=True,
                            name=None,
                            use_locking=False,
                            with_keras=False):
    algo = _ElasticSynchronousSGD(config, init_step, nccl, nccl_fusion)
    if with_keras:
        raise RuntimeError('TODO')
    else:
        return _create_kungfu_optimizer(optimizer, algo, name, use_locking)
