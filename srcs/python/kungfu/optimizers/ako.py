import sys

import tensorflow as tf

from kungfu.ops import global_step_modifier, ako_all_reduce, set_num_gradients
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
        self.akoPartitions = ako_partitions
        self.partitionIndices = None
        self._use_global_step = use_global_step
        if self._use_global_step:
            self._trained_steps = tf.Variable(tf.zeros([], tf.int32))
            self._modify_trained_steps = tf.assign(
                self._trained_steps, global_step_modifier(self._trained_steps))

    def __get_size(self, tensor):
        return tensor.shape.num_elements() * tensor.dtype.size

     # https://www8.cs.umu.se/kurser/TDBA77/VT06/algorithms/BOOK/BOOK2/NODE45.HTM
    def __reconstruct_partition(self, grads_and_vars, k, D):
        result = []
        n = len(D)
        k = k - 2
        while k >= 0:
            inner = []
            for i in range(D[n - 1][k] + 1, n + 1):
                inner.append(grads_and_vars[i])
            result.append(inner)
            n = D[n - 1][k]
            k -= 1

        inner = []
        for i in range(n + 1):
            inner.append(grads_and_vars[i])
        result.append(inner)
        result.reverse()
        return result

    def __partition_positions(self, grads_sizes, k):
        n = len(grads_sizes)
        # M[n][k] array of size n divided into k
        M = [[0 for i in range(k)] for j in range(n)]
        # D[n - 1][k - 1] separators
        D = [[0 for i in range(k - 1)] for j in range(n - 1)]

        M[0][0] = grads_sizes[0]
        # prefix sums
        for i in range(1, n):
            M[i][0] = M[i - 1][0] + grads_sizes[i]

        # init boundary condition
        for i in range(1, k):
            M[0][i] = grads_sizes[0]

        for i in range(1, n):
            for j in range(1, k):
                current_min = -1
                min_separator_pos = sys.maxsize
                for pos in range(i):
                    s = max(M[pos][j - 1], M[i][0] - M[pos][0])
                    if current_min < 0 or s < current_min:
                        current_min = s
                        min_separator_pos = pos
                M[i][j] = current_min
                D[i - 1][j - 1] = min_separator_pos
        return D

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        """Negotiate grad with peers, following flexible strategy."""

        def build_op():
            with tf.variable_scope('NegotiateGradients'):
                if self.partitionIndices is None:
                    # Get partition indices by size (runs once)
                    sizes = [self.__get_size(g) for g, _v in grads_and_vars_to_negotiate]
                    self.partitionIndices = self.__partition_positions(sizes, self.akoPartitions)

                # pair tensor name bucket id
                partitions = self.__reconstruct_partition(grads_and_vars_to_negotiate,  self.akoPartitions, self.partitionIndices)
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