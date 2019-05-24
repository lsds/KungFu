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


def _tensor_size(t):
    return t.shape.num_elements() * t.dtype.size

def _get_self_rank():
    import os
    return int(os.getenv('KUNGFU_TEST_SELF_RANK'))


def _get_num_peers():
    import json, os
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    return len(cluster_spec['Peers'])

def send_to(rank, t):
    return _op_lib.send_to(rank, t, input_tensor_name=t.name)

def request_vars(rank, variables):
    request_avg_ops = []
    for var in variables:
        request_op = _op_lib.request_var(rank, var_name=var.name, shape=var.shape, dtype=var.dtype)
        request_avg_ops.append(request_op)
    return request_avg_ops

def merge_received(t):
    return _op_lib.merge_received(t,
                                  input_tensor_name=t.name,
                                  shape=t.shape,
                                  dtype=t.dtype)

def broadcast(t):
    return _op_lib.broadcast(t)


def all_reduce(t):
    return _op_lib.all_reduce(t, input_tensor_name=t.name)


def all_reduce_gpu(t):
    return _op_lib.all_reduce_gpu(t, input_tensor_name=t.name)


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


def ako_p2p(gradients, fraction):
    import tensorflow as tf

    print("Constructing Ako P2P negotiator with budget fraction %f." % fraction)

    total_size = sum([_tensor_size(t) for t in gradients])
    budget = int(fraction * total_size)
    indexes, num_partitions = _bin_pack(
        dict((t.name, _tensor_size(t)) for t in gradients), budget)
    groups = [[] for _ in range(num_partitions)]
    for t in gradients:
        groups[indexes[t.name]].append(t)

    send_ops = []
    for dest_rank in range(_get_num_peers()):
        k = dest_rank % num_partitions
        group_k = groups[k]
        for g in group_k:
            send_op = send_to(dest_rank, g)
            send_ops.append(send_op)

    with tf.control_dependencies(send_ops):
        merged_grads = [merge_received(g) for g in gradients]
        return merged_grads



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
