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


def cpu_group_all_reduce_variance_monitor(grads, batch_small):
    import json, os
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    num_workers = len(cluster_spec['Peers'])

    import tensorflow as tf
    batch_big = batch_small * num_workers

    negotiated_grads = [all_reduce(t) for t in grads]

    global_variance_ops = []

    for i in range(len(grads)):
        G_big   = negotiated_grads[i]
        G_small = grads[i]       

        G_sq_small = tf.norm(G_small)
        G_sq_small = tf.square(G_sq_small)
        score_big  = batch_big * G_sq_small

        G_sq_big    = tf.norm(G_big)
        G_sq_big    = tf.square(G_sq_big)
        score_small = batch_small * G_sq_small

        G_biased = 1/(batch_big - batch_small) * (score_big - score_small)
        S_biased = 1/(1/batch_small - 1/batch_big) * (G_sq_small - G_sq_big)

        global_var_op = _op_lib.global_variance(G_biased, S_biased, input_tensor_name=grads[i].name)
        global_variance_ops.append(global_var_op)

    with tf.control_dependencies(global_variance_ops):
         return negotiated_grads


def get_gradient_noise_operators(batch_small, grads, negotiated_grads):
    import tensorflow as tf
    import json, os
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    num_workers = len(cluster_spec['Peers'])
    batch_big = batch_small * num_workers

    global_variance_ops = []
    for i in range(len(grads)):
        G_big   = negotiated_grads[i]
        G_small = grads[i]       

        G_sq_small = tf.norm(G_small)
        G_sq_small = tf.square(G_sq_small)
        score_big  = batch_big * G_sq_small

        G_sq_big    = tf.norm(G_big)
        G_sq_big    = tf.square(G_sq_big)
        score_small = batch_small * G_sq_small

        G_biased = 1/(batch_big - batch_small) * (score_big - score_small)
        S_biased = 1/(1/batch_small - 1/batch_big) * (G_sq_small - G_sq_big)

        global_var_op = _op_lib.gradient_noise(G_biased, S_biased, input_tensor_name=grads[i].name, alpha=0.8)
        global_variance_ops.append(global_var_op)
    return global_variance_ops

def build_controller_op(negotiated_grads_and_vars):
    return [(_op_lib.controller(negotiated_grad), var) for negotiated_grad, var in negotiated_grads_and_vars]