import os
import platform
import sysconfig
from ctypes import cdll

EXT_SUFFIX_KEY = 'SO'  # 'EXT_SUFFIX' does't work for python2


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
    return _op_lib.all_reduce(t, input_tensor_name=t.name)


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


def partial_exchange_group_all_reduce(ts,
                                      fraction=0.3,
                                      accumulate=False,
                                      average="none"):
    import math
    total_size = sum([t.shape.num_elements() * t.dtype.size for t in ts])
    print("Total Size of All Gradients: " + str(total_size))
    print("The fraction is: " + str(fraction))
    budget = int(math.floor(fraction * total_size))
    print("The bucket budget is: " + str(budget))
    return [
        partial_exchange_all_reduce(t, budget, len(ts), accumulate, average)
        for t in ts
    ]


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


def _bin_pack(sizes, budget):
    lst = list(reversed(sorted([(size, name)
                                for name, size in sizes.items()])))
    budget = max(budget, lst[0][0])
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


def gpu_partial_exchange_group_all_reduce_front_end_partitioning(
        ts, fraction=0.3, accumulate=False, average="none"):
    import math
    import tensorflow as tf
    total_size = sum([t.shape.num_elements() * t.dtype.size for t in ts])
    print("Total Size of All Gradients: " + str(total_size))
    print("The fraction is: " + str(fraction))
    budget = int(math.floor(fraction * total_size))
    indexes = _bin_pack(
        dict([(t.name, t.shape.num_elements() * t.dtype.size) for t in ts]),
        budget)
    print("The bucket budget is: " + str(budget))

    gs = tf.Variable(tf.zeros([], dtype=tf.int64))
    advance_gs = tf.assign(gs, gs + 1)
    num_partitions = len(set(indexes.values()))

    name_order = [(i, t.name) for i, t in enumerate(ts)]

    # Construct groups
    groups = [[] for _ in range(num_partitions)]
    for t in ts:
        groups[indexes[t.name]].append(t)

    # Start all groups
    c_to_t_map = dict()
    reordered_cond_ops = [0] * len(ts)
    for i, g in enumerate(groups):
        cond_op = tf.cond(
            tf.equal(tf.mod(gs, num_partitions),
                     i), lambda: gpu_group_all_reduce(g), lambda: g)
        for c, t in zip(cond_op, g):
            for j, t_init_name in name_order:
                if t_init_name == t.name:
                    reordered_cond_ops[j] = c

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
