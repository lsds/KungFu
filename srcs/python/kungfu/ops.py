import os
import platform
import sysconfig
from ctypes import cdll

EXT_SUFFIX_KEY = 'SO'  # 'EXT_SUFFIX' does't work for python2

import tensorflow as tf

from kungfu.helpers.ako_partitioner import AkoPartitioner
from kungfu.helpers.bin_pack_partitioner import BinPackPartitioner


def get_num_peers():
    import json, os
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    num_peers = len(cluster_spec['Peers'])
    return num_peers


def _load_op_lib(name):
    module_path = os.path.dirname(__file__)
    suffix = sysconfig.get_config_var(EXT_SUFFIX_KEY)
    filename = os.path.join(module_path, name + suffix)
    import tensorflow as tf
    return tf.load_op_library(filename)


def _load_init_lib(name):
    module_path = os.path.dirname(__file__)
    suffix = 'so' if platform.uname()[0] != 'Darwin' else 'dylib'
    filename = os.path.join(module_path, name + '.' + suffix)
    return cdll.LoadLibrary(filename)


def _load_and_init_op_lib():
    _op_lib = _load_op_lib('kungfu_tensorflow_ops')
    _init_lib = _load_init_lib('libkungfu_tensorflow_init')
    _init_lib.kungfu_tensorflow_init()
    has_gpu = False
    try:
        # FIXME: auto detect GPU support
        _init_lib.kungfu_tensorflow_init_gpu()
        has_gpu = True
    except Exception as e:
        print('kungfu_tensorflow_init_gpu FAILED: %s' % e)
    return _op_lib, has_gpu


_op_lib, _has_gpu = _load_and_init_op_lib()


def broadcast(t):
    return _op_lib.broadcast(t)


def all_reduce(t):
    return _op_lib.all_reduce(t, input_tensor_name=t.name[:-2])


def all_reduce_gpu(t):
    return _op_lib.all_reduce_gpu(t, input_tensor_name=t.name[:-2])


def ako_all_reduce(t, partition_id, num_partitions):
    return _op_lib.ako_negotiator(t, partition_id, num_partitions)


def global_variance(t):
    return _op_lib.global_variance(t)


def global_step_modifier(step):
    return _op_lib.global_step_modifier(step)


def set_num_gradients(n):
    return _op_lib.set_num_gradients(n)


def start_gpu_group(*args, **kwargs):
    return _op_lib.start_gpu_group(*args, **kwargs)


def _bin_pack(sizes, budget, adjust_budget=False):
    lst = list(reversed(sorted([(size, name)
                                for name, size in sizes.items()])))
    if adjust_budget:
        budget = max(budget, lst[0][0])
    else:
        raise RuntimeError("Budget is too small.")
    budgets = []
    indexes = dict()
    for size, name in lst:
        ok = False
        for i, b in enumerate(budgets):
            if b >= size:
                budgets[i] -= size
                indexes[name] = i
                ok = True
                break
        if not ok:
            budgets.append(budget - size)
            indexes[name] = len(budgets) - 1
    return indexes


def _bin_pack(sizes, budget, adjust_budget=False):
    lst = list(reversed(sorted([(size, name)
                                for name, size in sizes.items()])))
    if adjust_budget:
        budget = max(budget, lst[0][0])
    else:
        raise RuntimeError("Budget is too small.")
    budgets = []
    indexes = dict()
    for size, name in lst:
        ok = False
        for i, b in enumerate(budgets):
            if b >= size:
                budgets[i] -= size
                indexes[name] = i
                ok = True
                break
        if not ok:
            budgets.append(budget - size)
            indexes[name] = len(budgets) - 1
    return indexes


def _tensor_size(t):
    return t.shape.num_elements() * t.dtype.size


def _parse_schedule(schdule):
    # schedule is of the form
    # f1;e1;f2;e2;f3;e3
    tokens = schedule.split(";")
    epochs = [0] + [int(t) for t in tokens[1::2]]
    pairs = []
    for i in range(len(epochs) - 1):
        pairs.append((epochs[i], epochs[i + 1]))
    fractions = [float(t) for t in tokens[::2]]
    return zip(pairs, fractions)


def print_info(fraction, total_size, budget):
    print("The fraction is: " + str(fraction))
    print("Total Size of All Gradients: " + str(total_size))
    print("The bucket budget is: " + str(budget))


def adaptive_partial_exchange_with_cpu_allreduce(ts,
                                                 batch_size,
                                                 num_train,
                                                 schedule,
                                                 accumulate=False,
                                                 average="none"):
    print("Using piecewise partitioning schedule: " + schedule)
    parsed_schedule = _parse_schedule(schedule)
    import math
    import tensorflow as tf
    total_size = sum([_tensor_size(t) for t in ts])

    def build_partial_exchange_ops(fraction):
        budget = int(math.floor(fraction * total_size))
        indexes = _bin_pack(dict((t.name, _tensor_size(t)) for t in ts),
                            budget)
        print_info(fraction, total_size, budget)

        gs = tf.Variable(tf.zeros([], dtype=tf.int64))
        advance_gs = tf.assign(gs, gs + 1)

        num_partitions = len(set(indexes.values()))

        # Construct groups
        groups = [[] for _ in range(num_partitions)]
        for t in ts:
            groups[indexes[t.name]].append(t)

        # Start all groups
        reordered_cond_ops = [None] * len(ts)
        for i, partition in enumerate(groups):
            negotiated_partition = tf.cond(
                tf.equal(tf.mod(gs - 1, num_partitions), i), lambda:
                cpu_group_all_reduce(partition), lambda: partition)
            for negotiated_grad, grad in zip(negotiated_partition, partition):
                reordered_cond_ops[name_order[grad.name]] = negotiated_grad
        return reordered_cond_ops

    def tf_is_between_closed_open(x, left, right):
        l = tf.math.logical_or(tf.greater(x, left), tf.equal(x, left))
        r = tf.less(x, right)
        return tf.math.logical_and(l, r)

    with tf.control_dependencies([advance_gs]):
        initial_partitioning_ops = build_partial_exchange_ops(fraction)

        epoch = tf.div(gs, num_train / (batch_size * get_num_peers()))

        cond_ops = []
        for (pivot_epoch_l, pivot_epoch_r), pivot_fraction in parsed_schedule:
            cond = tf.cond(tf_is_between_closed_open(epoch, pivot_epoch_l, pivot_epoch_r), lambda: build_partial_exchange_ops(pivot_fraction), lambda: tf.no_op(name="No Op Dynamic Partitioning"))
            cond_ops.append(cond)
        return cond_ops


def cpu_group_all_reduce(ts):
    return [all_reduce(t) for t in ts]


def gpu_group_all_reduce(ts):
    names = [t.name[:-2] for t in ts]
    names = list(sorted(names))  # FIXME: use topsort
    import tensorflow as tf
    with tf.control_dependencies([
            start_gpu_group(names),
    ]):
        return [all_reduce_gpu(t) for t in ts]


def group_all_reduce(ts):
    # FIXME: auto determine device
    if _has_gpu:
        return gpu_group_all_reduce(ts)
    print('USING CPU GROUP ALL REDUCE')
    return cpu_group_all_reduce(ts)
