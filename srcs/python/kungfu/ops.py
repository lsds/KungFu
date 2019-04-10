import os
import platform
import sysconfig
from ctypes import cdll

import tensorflow as tf

def _load_op_lib(name):
    module_path = os.path.dirname(__file__)
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
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

from kungfu.helpers.ako_partitioner import AkoPartitioner

# Benchmark negotiator
def kungfu_strategy_negotiate(gradient_tensors, strategy='parallel', device='cpu', num_partitions=1):
    if strategy == 'parallel':
        if device == 'cpu':
           return cpu_group_all_reduce(gradient_tensors)
        elif device == 'gpu':
           return gpu_group_all_reduce(gradient_tensors)
        else:
           raise Exception('KungFu does not support the specified device type: ' + device)
    elif strategy == 'ako':
        partitioner      = AkoPartitioner(num_partitions)
        grads_and_vars_to_negotiate = [(grad, grad.name[:-2]) for grad in gradient_tensors]
        partitionIndices = partitioner.partition_positions(grads_and_vars_to_negotiate)
        partitions       = partitioner.reconstruct_partition(grads_and_vars_to_negotiate, partitionIndices)
        
        negotiated_grads = []
        for partition_id in range(len(partitions)):
            for grad, var in partitions[partition_id]:
                with tf.variable_scope('AkoMaybeNegotiatedGrad'):
                    # careful, ako_all_reduce may return a grad and a variable tuple
                    negotiated_grad = ako_all_reduce(
                                                      grad,
                                                      tf.constant([partition_id], dtype=tf.int32),
                                                      tf.constant([num_partitions], dtype=tf.int32))
                    # negotiated_grad = tf.clip_by_global_norm(negotiated_grad, 0.002)
                negotiated_grads.append(negotiated_grad)
        return negotiated_grads
    else:
        raise Exception('KungFu does not support the specified synchronization strategy: ' + strategy)

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
