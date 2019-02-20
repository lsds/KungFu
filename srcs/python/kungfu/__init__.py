import os
import sys
import sysconfig

import random
from functools import reduce

import tensorflow as tf

__all__ = [
    'SyncSGDOptimizer',
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


class KungFuOptimizer(tf.train.Optimizer):
    """An optimizer that would negotiate the gradients before apply it."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse=''):
        if name is None:
            name = "KungFuOptimizer{}".format(type(optimizer).__name__)
        super(KungFuOptimizer, self).__init__(
            name=name, use_locking=use_locking)

        self._optimizer = optimizer
        self._device_dense = device_dense
        self._device_sparse = device_sparse

        self._enable_set_num_gradients = True

    def _negotiate_grads_by_strategy(self, grads_and_vars):
        raise RuntimeError('Not implemented')
        # The subclass should implement this with its own negotiation strategy

    def _set_num_gradients(self, n):
        raise RuntimeError('Not implemented')

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients and negotiate with peers."""
        grads_and_vars = self._optimizer.compute_gradients(*args, **kwargs)
        grads_and_vars_to_negotiate = []
        for grad, var in grads_and_vars:
            if grad is not None:
                grads_and_vars_to_negotiate.append((grad, var))

        def build_op():
            # returns negotiated (gradient, variable) pairs
            return self._negotiate_grads_by_strategy(
                grads_and_vars_to_negotiate)

        if self._enable_set_num_gradients:
            n_grads = len(grads_and_vars_to_negotiate)
            with tf.control_dependencies([self._set_num_gradients(n_grads)]):
                return build_op()
        else:
            return build_op()

    # forward to the underlying optimizer

    def apply_gradients(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.apply_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.variables(*args, **kwargs)


class SyncSGDOptimizer(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""

    def __init__(
            self,
            optimizer,
            name=None,
            use_locking=False,
            strategy='plain',  # or ako
            ako_partitions=1,
            staleness=1,
            kickin_time=200,
            device_dense='',
            device_sparse='',
            use_global_step=True):
        super(SyncSGDOptimizer, self).__init__(optimizer, name, use_locking,
                                               device_dense, device_sparse)
        self._op_lib = lazy_load_op_lib()
        self.strategy = strategy

        if self.strategy == 'ako':
            self.accum_map = dict()
            self.staleness = staleness
            self.akoPartitions = ako_partitions
            self.kickinTime    =  kickin_time
            self.partitionIndices = None

        print('KungFu strategy: ' + strategy)
        if self.strategy == 'ako':
            print('KungFu staleness: ' + str(self.staleness))
            print('KungFu p partitions: ' + str(self.akoPartitions))
            print('KungFu kick in time: ' + str(self.kickinTime))        

        self._use_global_step = use_global_step
        if self._use_global_step:
            self._trained_steps = tf.Variable(tf.zeros([], tf.int32))
            self._modify_trained_steps = tf.assign(
                self._trained_steps,
                self._op_lib.global_step_modifier(self._trained_steps))

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

    def __get_size(self, tensor):
        return tensor.shape.num_elements() * tensor.dtype.size

    # create k partitions of each of grads_and_vars
    # bucket gradients such that the size in bytes in each bucket
    # is approximately equal
    # def partition_gradients(self, grads_and_vars, k):
    #     sizes = [self.__get_size(g) for g, _v in grads_and_vars]
    #     D = self.__partition_positions(sizes, k)
    #     return self.__reconstruct_partition(grads_and_vars, k, D)

    # Map is a dict from variable to queue of gradients
    def accumulate(self, grad, var, map, partitions):
        if var not in map:
            map[var] = [grad] #([grad], grad)
        else:
            # queue_gradiens, running_sum = map[var]
            queue_gradiens = map[var]
            #queue_gradiens.append((grad, tf.add_n([running_sum, grad])))
            queue_gradiens.append(grad)
            # Accumulate last #partitions gradients
            if len(queue_gradiens) > partitions:
                # Restore invariant
                #subtract_from_sum = queue_gradiens[0]
                queue_gradiens.pop(0)

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        """Negotiate grad with peers, following flexible strategy."""

        # def tensor_less(global_step, kickin_step):
        #     return tf.cond(tf.less(global_step, kickin_step), lambda: False, lambda: True)

        def build_op():
            with tf.variable_scope('NegotiatedGrad'):
                # global_step = tf.to_int32(self._op_lib.get_global_step(tf.zeros([], tf.int32)))
                if self.strategy == 'plain':
                    negotiated_grad_and_vars = []
                    for grad, var in grads_and_vars_to_negotiate:
                        negotiated_grad_and_vars.append(
                            (self._op_lib.all_reduce(grad), var))
                    return negotiated_grad_and_vars
                elif self.strategy == 'ako':
                    if self.partitionIndices is None:
                       # Get partition indices by size (runs once)
                       sizes = [self.__get_size(g) for g, _v in grads_and_vars_to_negotiate]
                       self.partitionIndices = self.__partition_positions(sizes, self.akoPartitions)
                    
                    partitions = self.__reconstruct_partition(grads_and_vars_to_negotiate,  self.akoPartitions, self.partitionIndices)
                    negotiated_grad_and_vars = []
                    for partition_id in range(len(partitions)):
                        for grad, var in partitions[partition_id]:
                            self.accumulate(grad, var, self.accum_map, self.akoPartitions)
                            # TODO: optimize, running sum
                            grad_accum = tf.add_n(self.accum_map[var]) / len(self.accum_map[var])
                            negotiated_grad_var = (self._op_lib.ako_negotiator(
                                                                grad_accum, 
                                                                grad,
                                                                tf.constant([partition_id], dtype=tf.int32),
                                                                tf.constant([self.akoPartitions], dtype=tf.int32),
                                                                tf.constant([self.kickinTime], dtype=tf.int32)),
                                                    var
                                                  )
                            negotiated_grad_and_vars.append(negotiated_grad_var)
                    return negotiated_grad_and_vars
                else:
                    raise RuntimeError('Strategy not implemented')

        if self._use_global_step:
            with tf.control_dependencies([self._modify_trained_steps]):
                return build_op()
        else:
            return build_op()

    def _set_num_gradients(self, n):
        return self._op_lib.set_num_gradients(tf.constant(n, tf.int32))
from kungfu.optimizers.core import lazy_load_op_lib


def distributed_variables_initializer():
    import tensorflow as tf
    _op_lib = lazy_load_op_lib()
    g = tf.get_default_graph()
    ops = []
    # TODO: auto inject tf.global_variables_initializer
    # with tf.control_dependencies([tf.global_variables_initializer()]):
    for v in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        ops.append(tf.assign(v, _op_lib.broadcast(v)))
    return ops
