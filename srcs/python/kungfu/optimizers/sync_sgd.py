import tensorflow as tf
from kungfu.ops import (broadcast, global_noise_scale, group_all_reduce,
                        group_nccl_all_reduce, peer_info)

from .core import KungFuOptimizer, defuse, fuse


class SyncSGDOptimizer(KungFuOptimizer):
    """SyncSGDOptimizer implements synchronous SGD.

    Synchronous SGD is explained in the following paper:
    Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
    https://arxiv.org/pdf/1706.02677

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "KungFuOptimizer" followed by the provided
        optimizer type.
      nccl:
        Optional flag for using NCCL to perform all-reduce.
      nccl_fusion:
        Optional flag to fuse all gradients before launch NCCL all-reduce.
        This is useful to amortise the cost of NCCL calls.
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.

    """
    def __init__(self,
                 optimizer,
                 nccl=False,
                 nccl_fusion=True,
                 name=None,
                 use_locking=False):
        super(SyncSGDOptimizer, self).__init__(optimizer,
                                               name,
                                               use_locking=use_locking)
        _rank, np = peer_info()
        # FIXME: use type of gradient
        self._num_workers = tf.cast(np, tf.float32)
        self._nccl = nccl
        self._nccl_fusion = nccl_fusion

    def apply_gradients(self, grads_and_vars, **kwargs):
        gradients, variables = list(zip(*grads_and_vars))

        if self._nccl:
            if self._nccl_fusion:
                fused_grad = fuse(gradients)
                summed_fused_gradients = group_nccl_all_reduce([fused_grad])
                summed_gradients = defuse(summed_fused_gradients[0],
                                          [g.shape for g in gradients])
            else:
                summed_gradients = group_nccl_all_reduce(gradients)
        else:
            summed_gradients = group_all_reduce(gradients)

        reduced_grads = [g / self._num_workers for g in summed_gradients]
        reduced_grads_and_vars = zip(reduced_grads, variables)
        return self._optimizer.apply_gradients(reduced_grads_and_vars,
                                               **kwargs)

    def distributed_initializer(self):
        ops = [tf.assign(v, broadcast(v)) for v in self.variables()]
        return tf.group(ops)


class SyncSGDWithGradVarianceOptimizer(KungFuOptimizer):
    """SyncSGDWithGradVarianceOptimizer monitors gradinet variance when performing synchronous SGD.

    You can find the defintion of variance of tensors here:
    https://en.wikipedia.org/wiki/Variance

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "KungFuOptimizer" followed by the provided
        optimizer type.
      monitor_interval:
        The interval of computing the variance for gradients.
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.

    """
    def __init__(self,
                 optimizer,
                 name=None,
                 monitor_interval=1,
                 use_locking=False):
        super(SyncSGDWithGradVarianceOptimizer,
              self).__init__(optimizer, name, use_locking=use_locking)
        _rank, np = peer_info()
        # FIXME: use type of gradient
        self._num_workers = tf.cast(np, tf.float32)
        self._step = tf.Variable(0, trainable=False, dtype=tf.int32)

        self._interval = monitor_interval
        self._summed_variance = None
        self._variances = None

    def _monitor(self, grads, reduced_grads):
        square_grads = [tf.square(g) for g in grads]
        summed_square_grads = group_all_reduce(square_grads)
        reduced_square_grads = [
            g / self._num_workers for g in summed_square_grads
        ]
        grad_variances = [
            square_grad - tf.square(grad)
            for square_grad, grad in zip(reduced_square_grads, reduced_grads)
        ]
        self._variances = [
            tf.norm(grad_variance) for grad_variance in grad_variances
        ]
        self._summed_variance = tf.reduce_sum(self._variances)
        print_op = tf.print('Sum of gradient variance:', self._summed_variance)

        with tf.control_dependencies([print_op]):
            return tf.no_op()

    def get_grad_variance(self):
        if self._variances == None or self._summed_variance == None:
            raise Exception(
                'Must be called after minimize() or apply_gradients()')
        return self._variances, self._summed_variance

    def apply_gradients(self, grads_and_vars, **kwargs):
        grads, variables = list(zip(*grads_and_vars))

        # Synchronisation logic
        summed_grads = group_all_reduce(grads)
        reduced_grads = [g / self._num_workers for g in summed_grads]

        # Monitoring logic
        monitor_grads_op = tf.cond(
            tf.equal(tf.mod(self._step, self._interval), 0),
            lambda: self._monitor(grads, reduced_grads), lambda: tf.no_op())

        with tf.control_dependencies([monitor_grads_op]):
            with tf.control_dependencies([tf.assign_add(self._step, 1)]):
                return self._optimizer.apply_gradients(
                    zip(reduced_grads, variables), **kwargs)

    def distributed_initializer(self):
        ops = [tf.assign(v, broadcast(v)) for v in self.variables()]
        return tf.group(ops)


class SyncSGDWithGradNoiseScaleOptimizer(KungFuOptimizer):
    """SyncSGDWithGradNoiseScaleOptimizer monitors gradinet noise scale when performing synchronous SGD.

    Gradient noise scale is proposed in:
    An Empirical Model of Large-Batch Training
    https://arxiv.org/abs/1812.06162

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      device_batch_size:
        The training batch size of the local device
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "KungFuOptimizer" followed by the provided
        optimizer type.
      monitor_interval:
        The interval of computing the variance for gradients.
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.

    """
    def __init__(self,
                 optimizer,
                 device_batch_size,
                 name=None,
                 monitor_interval=1,
                 use_locking=False):
        super(SyncSGDWithGradNoiseScaleOptimizer,
              self).__init__(optimizer, name, use_locking=use_locking)
        _rank, np = peer_info()
        # FIXME: use type of gradient
        self._num_workers = tf.cast(np, tf.float32)
        self._step = tf.Variable(0, trainable=False, dtype=tf.int32)

        self._interval = monitor_interval
        self._device_batch_size = tf.cast(device_batch_size, dtype=tf.float32)
        self._global_batch_size = self._device_batch_size * self._num_workers
        self._noise_op = None

    def _monitor(self, grads, reduced_grads):
        self._noise_op = global_noise_scale(self._device_batch_size,
                                            self._global_batch_size,
                                            fuse(grads), fuse(reduced_grads))

        print_op = tf.print('Gradient Noise Scale:', self._noise_op)

        with tf.control_dependencies([print_op]):
            return tf.no_op()

    def get_grad_noise_scale(self):
        if self._noise_op == None:
            raise Exception(
                'Must be called after minimize() or apply_gradients()')
        return self._noise_op

    def apply_gradients(self, grads_and_vars, **kwargs):
        grads, variables = list(zip(*grads_and_vars))

        # Synchronisation logic
        summed_grads = group_all_reduce(grads)
        reduced_grads = [g / self._num_workers for g in summed_grads]

        # Monitoring logic
        monitor_grads_op = tf.cond(
            tf.equal(tf.mod(self._step, self._interval), 0),
            lambda: self._monitor(grads, reduced_grads), lambda: tf.no_op())

        with tf.control_dependencies([monitor_grads_op]):
            with tf.control_dependencies([tf.assign_add(self._step, 1)]):
                return self._optimizer.apply_gradients(
                    zip(reduced_grads, variables), **kwargs)

    def distributed_initializer(self):
        ops = [tf.assign(v, broadcast(v)) for v in self.variables()]
        return tf.group(ops)
