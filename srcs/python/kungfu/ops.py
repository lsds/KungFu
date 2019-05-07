from __future__ import print_function

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
########################## Global Gradient Variance ################
def cpu_group_all_reduce_global_variance(grads):
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    num_workers = len(cluster_spec['Peers'])
    if num_workers == 0:
        raise "Cluster spec KUNGFU_CLUSTER_SPEC is invalid"

    negotiated_grads = [all_reduce(t) for t in grads]

    # Compute negotiated_global_variances
    # sum (grad - negotiated_grad/#num_workers)^2
    for i in range(len(grads)):
        g  = grads[i]
        ng = negotiated_grads[i] 
        ng = tf.div(ng, num_workers)
        # TODO

########################### Gradient Noise #########################

def concat_gradients(grads):
    import tensorflow as tf
    reshaped_grads = []
    for t in grads:
        reshaped_grad = tf.reshape(t, [-1])
        reshaped_grads.append(reshaped_grad)
    
    stacked = tf.concat(reshaped_grads, -1)
    flat_all = tf.reshape(stacked, [-1])
    print(flat_all)
    return flat_all


def global_noise_tensorboard(total):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    import tensorflow as tf
    with tf.name_scope('summaries'):
        tf.summary.scalar('total', total)


def cpu_group_all_reduce_variance_monitor(grads, batch_small):
    negotiated_grads = [all_reduce(t) for t in grads]

    import tensorflow as tf

    concat_grad            = concat_gradients(grads)
    concat_negotiated_grad = concat_gradients(negotiated_grads)

    print(concat_grad.shape)
    print(concat_negotiate_grad.shape)

    noise_op = get_global_gradient_noise_operator(batch_small, concat_grad, concat_negotiated_grad)
    print_op_total = tf.Print(noise_op, [noise_op], message="Total Gradient Noise at current iteration") 
    global_noise_tensorboard(noise_op)

    #merged = tf.summary.merge_all() 

    with tf.control_dependencies([noise_op, print_op_total]): # add merged
        return [_op_lib.controller(negotiated_grad) for negotiated_grad in negotiated_grads]

def get_global_gradient_noise_operator(batch_small, concat_grad, concat_negotiated_grad):
    import tensorflow as tf
    import json, os
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    num_workers = len(cluster_spec['Peers'])
    if num_workers == 0:
        raise "Cluster spec KUNGFU_CLUSTER_SPEC is invalid"
    batch_big = batch_small * num_workers

    G_big = concat_negotiated_grad
    ## Take average over workers
    G_big = tf.div(concat_negotiated_grad, num_workers) 
    
    G_small = concat_grad     

    G_sq_small = tf.norm(G_small)
    G_sq_small = tf.square(G_sq_small)
    score_big  = batch_big * G_sq_small

    G_sq_big    = tf.norm(G_big)
    G_sq_big    = tf.square(G_sq_big)
    score_small = batch_small * G_sq_small

    G_biased = 1/(batch_big - batch_small) * (score_big - score_small)
    S_biased = 1/(1/batch_small - 1/batch_big) * (G_sq_small - G_sq_big)

    global_noise_op = _op_lib.gradient_noise(G_biased, S_biased, input_tensor_name="ConcatGradientNoise", alpha=0.6)

    return global_noise_op

def build_controller_op(negotiated_grads_and_vars):
    return [(_op_lib.controller(negotiated_grad), var) for negotiated_grad, var in negotiated_grads_and_vars]

