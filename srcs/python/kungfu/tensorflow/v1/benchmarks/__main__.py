"""
Usage:
    kungfu-run -q -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method CPU
    kungfu-run -q -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method NCCL
    kungfu-run -q -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method NCCL+CPU
    mpirun -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method HOROVOD
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from kungfu._utils import measure, one_based_range
from kungfu.python import _get_cuda_index
from kungfu.tensorflow.ops import (current_cluster_size, current_rank,
                                   group_all_reduce, group_nccl_all_reduce)
from kungfu.tensorflow.ops.collective import group_hierarchical_nccl_all_reduce
from kungfu.tensorflow.v1.benchmarks import model_sizes
from kungfu.tensorflow.v1.helpers.utils import show_rate, show_size
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


def _tensor_size(t):
    return t.shape.num_elements() * t.dtype.size


def hvd_init():
    import horovod.tensorflow as hvd
    hvd.init()


def hvd_group_all_reduce(ts):
    import horovod.tensorflow as hvd
    return [hvd.allreduce(t, average=False) for t in ts]


def get_cluster_size(method):
    if method == 'HOROVOD':
        import horovod.tensorflow as hvd
        return hvd.size()
    else:
        return current_cluster_size()


def get_rank(method):
    if method == 'HOROVOD':
        import horovod.tensorflow as hvd
        return hvd.rank()
    else:
        return current_rank()


_group_all_reduce_func = {
    'CPU': group_all_reduce,
    'NCCL': group_nccl_all_reduce,
    'NCCL+CPU': group_hierarchical_nccl_all_reduce,
    'HOROVOD': hvd_group_all_reduce,
}

_model_sizes = {
    'ResNet50': model_sizes.resnet50_imagenet,
    'VGG16': model_sizes.vgg16_imagenet,
    'BERT': model_sizes.bert,
}


def _config(method):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if method == 'HOROVOD':
        import horovod.tensorflow as hvd
        config.gpu_options.visible_device_list = str(hvd.local_rank())
    else:
        config.gpu_options.visible_device_list = str(_get_cuda_index())
    return config


def _rank(method):
    if method == 'HOROVOD':
        import horovod.tensorflow as hvd
        return hvd.rank()
    else:
        return current_rank()


def parse_args():
    p = argparse.ArgumentParser(description='Perf Benchmarks.')
    p.add_argument('--model',
                   type=str,
                   default='ResNet50',
                   help='ResNet50 | VGG16 | BERT')
    p.add_argument('--method',
                   type=str,
                   default='CPU',
                   help='CPU | NCCL | HOROVOD')
    p.add_argument('--fuse', action='store_true', default=False, help='')
    p.add_argument('--max-count', type=int, default=0, help='max grad count')
    p.add_argument('--steps',
                   type=int,
                   default=10,
                   help='number of steps to run')
    p.add_argument('--warmup-steps',
                   type=int,
                   default=5,
                   help='number of warmup steps')
    return p.parse_args()


def log_detailed_result(value, error, attrs):
    import json
    attr_str = json.dumps(attrs, separators=(',', ':'))
    # grep -o RESULT.* *.log
    unit = 'GiB/s'
    print('RESULT: %f +-%f (%s) %s' % (value, error, unit, attr_str))


def log_final_result(values, args):
    attrs = {
        'method': args.method,
        'np': get_cluster_size(args.method),
        'model': args.model,
        'fuse': args.fuse,
    }
    values = np.array(values)
    if args.method != 'HOROVOD':
        attrs['strategy'] = os.getenv('KUNGFU_ALLREDUCE_STRATEGY')
        attrs['nvlink'] = os.getenv('KUNGFU_ALLOW_NVLINK')
    log_detailed_result(values.mean(), 1.96 * values.std(), attrs)


def all_reduce_benchmark(sizes, dtype, args):
    rank = _rank(args.method)

    def log(msg):
        if rank == 0:
            print(msg)

    xs = [tf.Variable(tf.ones([n], dtype)) for n in sizes]
    tot_size = sum(_tensor_size(x) for x in xs)
    np = get_cluster_size(args.method)
    multiplier = 4 * (np - 1)
    log('all reduce %d tensors of total size: %s among %d peers, using %s' %
        (len(sizes), show_size(tot_size), np, args.method))

    ys = _group_all_reduce_func[args.method](xs)

    init = tf.global_variables_initializer()

    values = []
    with tf.Session(config=_config(args.method)) as sess:
        duration, _ = measure(lambda: sess.run(init))
        log('tensorflow init took %.fs' % (duration))

        for step in one_based_range(args.warmup_steps):
            duration, _ = measure(lambda: sess.run(ys))
            log('warmup step %d, took %.2fs, equivalent data rate: %s' %
                (step, duration, show_rate(tot_size * multiplier, duration)))

        for step in one_based_range(args.steps):
            duration, _ = measure(lambda: sess.run(ys))
            gi = 1024 * 1024 * 1024
            values.append(tot_size * multiplier / gi / duration)
            log('step %d, took %.2fs, equivalent data rate: %s' %
                (step, duration, show_rate(tot_size * multiplier, duration)))

    if get_rank(args.method) == 0:
        log_final_result(values, args)


def main(_):
    args = parse_args()
    if args.method == 'HOROVOD':
        hvd_init()
    dtype = tf.float32
    sizes = _model_sizes[args.model]
    if args.fuse:
        sizes = [sum(sizes)]
    if args.max_count > 0 and len(sizes) > args.max_count:
        sizes = sizes[:args.max_count]
    all_reduce_benchmark(sizes, dtype, args)


if __name__ == "__main__":
    main(sys.argv)
