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


def _tensor_size(t):
    return t.shape.num_elements() * t.dtype.size


def send_to(rank, t):
    return _op_lib.send_to(rank, t, input_tensor_name=t.name)


def merge_received(t):
    return _op_lib.merge_received(t,
                                  input_tensor_name=t.name,
                                  shape=t.shape,
                                  dtype=t.dtype)


def broadcast(t):
    return _op_lib.broadcast(t)


def all_reduce(t):
    return _op_lib.all_reduce(t, input_tensor_name=t.name)

# Based on Andrei - Octavian Brabete
def partial_exchange_all_reduce(t, budget, count_gradients, accumulate, average, fraction, find_epoch_denominator):
    # Take full gradient name for unicity
    tensor_size = t.shape.num_elements() * t.dtype.size
    if accumulate:
        import json, os
        cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
        num_peers = len(cluster_spec['Peers'])
        return _op_lib.partial_accumulating_negotiator(t, input_tensor_name=t.name, budget=budget, 
                                                       tensor_size=tensor_size, count_gradients=count_gradients, num_peers=num_peers, average=average)
    else:
        return _op_lib.partial_negotiator(t, input_tensor_name=t.name, budget=budget, 
                                          tensor_size=tensor_size, count_gradients=count_gradients, 
                                          find_epoch_denominator=find_epoch_denominator, fraction=fraction)

def partial_exchange_all_reduce_with_schedule(t, budget, total_size, count_gradients, steps, fractions):
    # Take full gradient name for unicity
    tensor_size = t.shape.num_elements() * t.dtype.size
    print(str(t))
    print(str(t.name))
    print(str(budget)) # zero
    print(str(tensor_size))
    print(str(tensor_size))
    print(str(count_gradients))
    print(str(steps))
    print(str(fractions))
    return _op_lib.partial_negotiator_with_schedule(t, total_size=total_size, input_tensor_name=t.name, budget=budget, 
                                        tensor_size=tensor_size, count_gradients=count_gradients, 
                                        steps=steps, fractions=fractions,
                                        fraction=fractions[0])


# Based on Guo Li
def partial_exchange_all_reduce_front_end_partitioning(t, index, partitions, accumulate=False, average=False):
    # Take full gradient name for unicity
    tensor_size = t.shape.num_elements() * t.dtype.size
    return _op_lib.partial_negotiator_front_end_partitioning(t, index=index, partitions=partitions)

def all_reduce_gpu(t):
    return _op_lib.all_reduce_gpu(t, input_tensor_name=t.name)

def ako_all_reduce(t, partition_id, num_partitions):
    return _op_lib.ako_negotiator(t, partition_id, num_partitions)

def global_variance(t):
    return _op_lib.global_variance(t)


def global_step_modifier(step):
    print('global_step_modifier is deprecated and will be removed soon')
    return _op_lib.global_step_modifier(step)


def set_num_gradients(n):
    print('set_num_gradients is deprecated and will be removed soon')
    return _op_lib.set_num_gradients(n)


def start_gpu_group(*args, **kwargs):
    return _op_lib.start_gpu_group(*args, **kwargs)


# Based on Andrei-Octavian Brabete, Dynamic Partitioning done within C++ operator
def partial_exchange_group_all_reduce_with_schedule(ts, batch_size, num_train, schedule):
    import math

    steps, fractions = _parse_schedule(schedule, batch_size, num_train)
    steps[0] = 1
    fraction = fractions[0]

    total_size = sum([t.shape.num_elements() * t.dtype.size for t in ts])
    print("Total Size of All Gradients: " + str(total_size))
    print("The fraction is: " + str(fraction))
    budget = int(math.floor(fraction * total_size))
    print("The bucket budget is: " + str(budget))

    trained_steps_op = tf.Variable(tf.zeros([], tf.int32))
    modify_trained_steps_op = tf.assign(
            trained_steps_op, global_step_modifier(trained_steps_op))

    
    with tf.control_dependencies([modify_trained_steps_op]):
        return [partial_exchange_all_reduce_with_schedule(t, budget, total_size, len(ts), steps, fractions) for t in ts]



# Based on Andrei-Octavian Brabete, Partitioning done within C++ operator
def partial_exchange_group_all_reduce(ts, batch_size, num_train, fraction=0.3, accumulate=False, average="none"):
    import math
    total_size = sum([t.shape.num_elements() * t.dtype.size for t in ts])
    print("Total Size of All Gradients: " + str(total_size))
    print("The fraction is: " + str(fraction))
    budget = int(math.floor(fraction * total_size))
    print("The bucket budget is: " + str(budget))

    trained_steps_op = tf.Variable(tf.zeros([], tf.int32))
    modify_trained_steps_op = tf.assign(
            trained_steps_op, global_step_modifier(trained_steps_op))

    with tf.control_dependencies([modify_trained_steps_op]):
        return [partial_exchange_all_reduce(t, budget, len(ts), accumulate, average, fraction, num_train / (batch_size * get_num_peers())) for t in ts]


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
    steps, fractions = list(steps), list(fractions)

    print("Steps: " + str(steps))
    print("Fractions: " + str(fractions))
    return steps, fractions


# Based on Guo Li, Partitioning in python
def partial_exchange_group_all_reduce_front_end_partitioning(ts, fraction=0.3, accumulate=False, average="none"):
    import math
    total_size = sum([t.shape.num_elements() * t.dtype.size for t in ts])
    print("Total Size of All Gradients: " + str(total_size))
    print("The fraction is: " + str(fraction))
    binpacker = BinPackPartitioner()
    budget = int(math.floor(fraction * total_size))
    indexes = binpacker.bin_pack(dict([(t.name, t.shape.num_elements() * t.dtype.size) for t in ts]), budget)
    print("The bucket budget is: " + str(budget))


    trained_steps_op = tf.Variable(tf.zeros([], tf.int32))
    modify_trained_steps_op = tf.assign(
            trained_steps_op, global_step_modifier(trained_steps_op))

    with tf.control_dependencies([modify_trained_steps_op]):
        return [
        # pass indexes[t.name] instead of budget
            partial_exchange_all_reduce_front_end_partitioning(t, index=indexes[t.name], partitions=len(set(indexes.values()))) for t in ts
        ]


def ako_group_all_reduce(gradient_tensors, num_partitions=1):
    partitioner      = AkoPartitioner(num_partitions)
    grads_and_vars_to_negotiate = [(grad, grad.name) for grad in gradient_tensors]
    partitionIndices = partitioner.partition_positions(grads_and_vars_to_negotiate)
    partitions       = partitioner.reconstruct_partition(grads_and_vars_to_negotiate, partitionIndices)
    
    # Print info
    partitioner.print_gradient_info(grads_and_vars_to_negotiate)
    partitioner.print_partition_info(num_partitions, partitions)
    
    negotiated_grads = []
    for partition_id in range(len(partitions)):
        for grad, var in partitions[partition_id]:
            with tf.variable_scope('AkoMaybeNegotiatedGrad'):
                negotiated_grad = ako_all_reduce(grad,tf.constant([partition_id], dtype=tf.int32),
                                                 tf.constant([num_partitions], dtype=tf.int32))
            negotiated_grads.append(negotiated_grad)
    return negotiated_grads

def compute_partitions(fraction, ts, total_size, tensor_partition_idx_vars,
                       num_partitions_var):
    import math
    import tensorflow as tf
    budget = int(math.floor(fraction * total_size))
    indexes, new_num_partitions = _bin_pack(
        dict((t.name, _tensor_size(t)) for t in ts), budget)
    print("Fraction: " + str(fraction))
    print("Size indices: " + str(len(indexes.values())))

    assign_partitions = tf.assign(num_partitions_var, new_num_partitions)

    assign_idx_vars = []
    for k in indexes.keys():
        # k is tensor name
        assign_idx_var = tf.assign(tensor_partition_idx_vars[k], indexes[k])
        assign_idx_vars.append(assign_idx_var)
    with tf.control_dependencies(assign_idx_vars + [assign_partitions]):
        return tf.constant(True, dtype=tf.bool)


def adaptive_partial_exchange_with_cpu_allreduce(ts,
                                                 batch_size,
                                                 num_train,
                                                 schedule,
                                                 accumulate=False,
                                                 average="none"):
    import tensorflow as tf
    print("Using piecewise partitioning schedule: " + schedule)
    steps, fractions = _parse_schedule(schedule, int(batch_size),
                                       int(num_train))

    total_size = sum([_tensor_size(t) for t in ts])
    global_step = tf.Variable(
        tf.zeros([], dtype=tf.int64)
    )  # tf.train.get_or_create_global_step(graph=tf.get_default_graph())
    increment_global_step_op = tf.assign(global_step, global_step + 1)

    # Dynamic partition info
    tensor_partition_idx_vars = dict(
        (t.name, tf.Variable(tf.ones([], dtype=tf.int64))) for t in ts)
    num_partitions_var = tf.Variable(tf.ones([], dtype=tf.int64))

    # Reverse both
    steps = steps[::-1]
    fractions = fractions[::-1]

    cases = []
    for i in range(len(steps)):
        cases.append((tf.greater_equal(global_step - 1, steps[i]),
                      lambda frac=fractions[i]: compute_partitions(
                          frac, ts, total_size, tensor_partition_idx_vars,
                          num_partitions_var)))

    bin_pack_case = tf.case(cases,
                            exclusive=False,
                            default=lambda: tf.constant(True, dtype=tf.bool))

    with tf.control_dependencies([bin_pack_case]):
        partial_negotiated_ts = []
        for tensor in ts:
            partition_idx_var = tensor_partition_idx_vars[tensor.name]
            mod_op = tf.mod(global_step - 1, num_partitions_var)
            equal_op = tf.equal(mod_op, partition_idx_var)

            negotiated_grad = tf.cond(
                equal_op,
                lambda tensor=tensor, partition_idx_var=partition_idx_var,
                num_partitions_var=num_partitions_var: all_reduce(tensor),
                lambda tensor=tensor: tf.identity(tensor))
            partial_negotiated_ts.append(negotiated_grad)

        with tf.control_dependencies([increment_global_step_op]):
            return [tf.identity(pnt) for pnt in partial_negotiated_ts]


def cpu_group_all_reduce(ts):
    return [all_reduce(t) for t in ts]


def gpu_group_all_reduce(ts):
    names = [t.name for t in ts]
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


def _bin_pack(sizes, budget, adjust_budget=False):
    lst = list(reversed(sorted([(size, name)
                                for name, size in sizes.items()])))
    if adjust_budget:
        budget = max(budget, lst[0][0])
    else:
        if lst[0][0] > budget:
            print("Suggested budget fraction is %f" %
                  (lst[0][0] / sum([s[1] for s in sizes.items()])))
            raise RuntimeError("Budget is too small %f. Largest tensor is %f" %
                               (budget, lst[0][0]))

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
    return indexes, len(budgets)


def partial_exchange_with_gpu_allreduce(ts,
                                        fraction=0.3,
                                        accumulate=False,
                                        average="none"):
    import math
    import tensorflow as tf
    total_size = sum([_tensor_size(t) for t in ts])
    print("Total Size of All Gradients: " + str(total_size))
    print("The fraction is: " + str(fraction))
    budget = int(math.floor(fraction * total_size))
    indexes, num_partitions = _bin_pack(
        dict((t.name, _tensor_size(t)) for t in ts), budget)
    print("The bucket budget is: " + str(budget))

    gs = tf.Variable(tf.zeros([], dtype=tf.int64))
    advance_gs = tf.assign(gs, gs + 1)

    name_order = dict((t.name, i) for i, t in enumerate(ts))

    # Construct groups
    groups = [[] for _ in range(num_partitions)]
    for t in ts:
        groups[indexes[t.name]].append(t)

    # Start all groups
    reordered_cond_ops = [None] * len(ts)
    for i, partition in enumerate(groups):
        negotiated_partition = tf.cond(
            tf.equal(tf.mod(gs - 1, num_partitions), i),
            lambda partition=partition: gpu_group_all_reduce(partition),
            lambda partition=partition: partition)
        for negotiated_grad, grad in zip(negotiated_partition, partition):
            reordered_cond_ops[name_order[grad.name]] = negotiated_grad

    with tf.control_dependencies([advance_gs]):
        return reordered_cond_ops


def _concat(ts):
    import tensorflow as tf
    return tf.concat([tf.reshape(t, [-1]) for t in ts], -1)


def cpu_group_all_reduce_variance_monitor(grads, batch_small):
    import tensorflow as tf
    negotiated_grads = [all_reduce(t) for t in grads]
    noise_op = get_global_gradient_noise_operator(batch_small, _concat(grads),
                                                  _concat(negotiated_grads))
    with tf.control_dependencies([noise_op]):
        return [
            _op_lib.controller(negotiated_grad)
            for negotiated_grad in negotiated_grads
        ]


def get_global_gradient_noise_operator(batch_small, concat_grad,
                                       concat_negotiated_grad):
    import tensorflow as tf
    import json, os
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    num_workers = len(cluster_spec['Peers'])
    if num_workers == 0:
        raise "Cluster spec KUNGFU_CLUSTER_SPEC is invalid"
    batch_big = batch_small * num_workers

    # Take average over workers
    G_big = tf.div(concat_negotiated_grad, num_workers)
    G_small = concat_grad

    G_sq_small = tf.square(tf.norm(G_small))
    G_sq_big = tf.square(tf.norm(G_big))

    G_biased = 1 / (batch_big - batch_small) * (batch_big * G_sq_big -
                                                batch_small * G_sq_small)
    S_biased = 1 / (1 / batch_small - 1 / batch_big) * (G_sq_small - G_sq_big)

    global_noise_op = _op_lib.gradient_noise(G_biased, S_biased, alpha=0.6)

    return global_noise_op
