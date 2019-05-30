import sys

import tensorflow as tf

from kungfu.ops import global_step_modifier, ako_all_reduce, set_num_gradients
from kungfu.helpers.ako_partitioner import AkoPartitioner
from .core import KungFuOptimizer


class AkoOptimizer(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 ako_partitions=1,
                 device_dense='',
                 device_sparse='',
                 use_global_step=True):
        super(AkoOptimizer, self).__init__(optimizer, name, use_locking,
                                           device_dense, device_sparse)
        self.partitioner = AkoPartitioner(ako_partitions)
        self.akoPartitions = ako_partitions
        self.partitionIndices = None
        self._use_global_step = use_global_step
        if self._use_global_step:
            self._trained_steps = tf.Variable(tf.zeros([], tf.int32))
            self._modify_trained_steps = tf.assign(
                self._trained_steps, global_step_modifier(self._trained_steps))

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        """Negotiate grad with peers, following flexible strategy."""

        def build_op():
            with tf.variable_scope('NegotiateGradients'):
                if self.partitionIndices is None:
                    # Get partition indices by size (runs once)
                    self.partitionIndices = self.partitioner.partition_positions(grads_and_vars_to_negotiate)
                # pair tensor name bucket id
                partitions = self.partitioner.reconstruct_partition(grads_and_vars_to_negotiate, self.partitionIndices)
                
                self.partitioner.print_gradient_info(grads_and_vars_to_negotiate)
                self.partitioner.print_partition_info(self.akoPartitions, partitions)
            
                negotiated_grad_and_vars = []
                for partition_id in range(len(partitions)):
                    for grad, var in partitions[partition_id]:
                        with tf.variable_scope('AkoMaybeNegotiatedGrad'):
                            negotiated_grad_var = (ako_all_reduce(
                                                                grad,
                                                                tf.constant([partition_id], dtype=tf.int32),
                                                                tf.constant([self.akoPartitions], dtype=tf.int32)),
                                                    var
                                                    )
                        negotiated_grad_and_vars.append(negotiated_grad_var)
                return negotiated_grad_and_vars

        if self._use_global_step:
            with tf.control_dependencies([self._modify_trained_steps]):
                return build_op()
        else:
            return build_op()

    def _set_num_gradients(self, n):
        return set_num_gradients(tf.constant(n, tf.int32))