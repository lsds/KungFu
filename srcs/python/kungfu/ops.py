from __future__ import print_function

import os
import platform
import sysconfig
from ctypes import cdll

import sys

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


def all_reduce(t, partition_id_var, num_partitions_var):
    return _op_lib.all_reduce(t, partition_id_var, num_partitions_var, input_tensor_name=t.name)


def all_reduce_gpu(t):
    return _op_lib.all_reduce_gpu(t, input_tensor_name=t.name)


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

    max_size = lst[0][0]
    if adjust_budget:
        budget = max(budget, max_size)
    else:
        if budget < max_size:
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


def _print_info(fraction, total_size, budget):
    print("The fraction is: " + str(fraction))
    print("Total Size of All Gradients: " + str(total_size))
    print("The bucket budget is: " + str(budget))


def _parse_schedule(schedule, batch_size, num_train):
    # schedule is of the form
    # f1;e1;f2;e2;f3;e3
    tokens = schedule.split(",")
    print("Num train: " + str(num_train))
    print("Batch size: " + str(batch_size))
    to_gs = lambda epoch: int(epoch * num_train / (batch_size * get_num_peers(
    )))
    pairs = [(to_gs(int(t.split(":")[0])), float(t.split(":")[1]))
             for t in tokens]
    steps, fractions = zip(*pairs)

    print("Steps: " + str(steps))
    print("Fractions: " + str(fractions))
    return steps, fractions


def adaptive_partial_exchange_with_cpu_allreduce(ts,
                                                 batch_size,
                                                 num_train,
                                                 schedule,
                                                 accumulate=False,
                                                 average="none"):
    print("Using piecewise partitioning schedule: " + schedule)
    steps, fractions = _parse_schedule(schedule, int(batch_size),
                                       int(num_train))
    import math
    import tensorflow as tf

    total_size = sum([_tensor_size(t) for t in ts])
    global_step = tf.Variable(tf.zeros([], dtype=tf.int64)) #  tf.train.get_or_create_global_step(graph=tf.get_default_graph())
    increment_global_step_op = tf.assign(global_step, global_step + 1)

    print_global_step = tf.Print(global_step, [global_step], message="Global step")

    tensor_partition_idx_vars = dict(
        (t.name, tf.Variable(tf.ones([], dtype=tf.int64))) for t in ts)
    num_partitions_var = tf.Variable(tf.ones([], dtype=tf.int64))

    def compute_partitions(fraction, tensor_partition_idx_vars,
                           num_partitions_var):
        budget = int(math.floor(fraction * total_size))
        indexes = _bin_pack(dict((t.name, _tensor_size(t)) for t in ts),
                            budget)
        print("Fraction: " + str(fraction))
        print("Size indices: " + str(len(indexes.values())))
        new_num_partitions = len(indexes.values())

        assign_partitions = tf.assign(num_partitions_var, new_num_partitions)

        with tf.control_dependencies([num_partitions_var, assign_partitions]):
            for k in indexes.keys():
                assign_idx_var = tf.assign(tensor_partition_idx_vars[k], indexes[k])
                with tf.control_dependencies([tensor_partition_idx_vars[k], assign_idx_var]):
                     pass
        return tf.constant(True, dtype=tf.bool)

    cases = []
    for i in range(len(steps)):
        cases.append((tf.equal(global_step, steps[i]), 
        lambda frac=fractions[i]: compute_partitions(frac, tensor_partition_idx_vars, num_partitions_var)
                    ))
    bin_pack_case = tf.case(cases, exclusive=False, default=lambda: tf.constant(True, dtype=tf.bool))

    print_partitions = tf.Print(num_partitions_var, [num_partitions_var], message="Partition print: ")

    with tf.control_dependencies([bin_pack_case] + [global_step, increment_global_step_op, print_global_step] + [print_partitions]):
        partial_negotiated_ts = []
        for tensor in ts:
            partition_idx_var = tensor_partition_idx_vars[tensor.name]
            negotiated_grad = tf.cond(
                tf.equal(tf.mod(global_step, num_partitions_var), partition_idx_var),
                lambda: all_reduce(tensor, partition_idx_var, num_partitions_var), lambda: tensor)
            partial_negotiated_ts.append(negotiated_grad)

    with tf.control_dependencies(partial_negotiated_ts):
        return partial_negotiated_ts


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
