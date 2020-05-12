"""
Usage:
    kungfu-run -q -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method CPU
    kungfu-run -q -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method NCCL
    mpirun -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method HOROVOD
"""

import argparse
import sys

import tensorflow as tf
from kungfu._utils import measure, one_based_range, map_maybe
from kungfu.ext import _get_cuda_index, change_strategy
from kungfu.tensorflow.ops import (current_cluster_size, current_rank,
                                   group_all_reduce, group_nccl_all_reduce,
                                   monitored_all_reduce)
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


_group_all_reduce_func = {
    'CPU': group_all_reduce,
    'NCCL': group_nccl_all_reduce,
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

    p.add_argument('--adapt',
                   type=bool,
                   default=False,
                   help='True | False')
    return p.parse_args()


def all_reduce_benchmark(sizes, dtype=tf.float32, method='CPU', adapt=False):
    rank = _rank(method)

    def log(msg):
        if rank == 0:
            print(msg)

    xs = [tf.Variable(tf.ones([n], dtype)) for n in sizes]
    tot_size = sum(_tensor_size(x) for x in xs)
    np = get_cluster_size(method)
    multiplier = 4 * (np - 1)
    log('all reduce %d tensors of total size: %s among %d peers, using %s' %
        (len(sizes), show_size(tot_size), np, method))

    # ys = _group_all_reduce_func[method](xs)
    # ys = map_maybe(monitored_all_reduce, xs)
    ys = [monitored_all_reduce(x, []) for x in xs]

    init = tf.global_variables_initializer()

    warmup_steps = 5
    bench_steps = 10
    changed = False

    with tf.Session(config=_config(method)) as sess:
        duration, _ = measure(lambda: sess.run(init))
        log('tensorflow init took %.fs' % (duration))

        for step in one_based_range(warmup_steps):
            duration, _ = measure(lambda: sess.run(ys))
            log('warmup step %d, took %.2fs, equivalent data rate: %s' %
                (step, duration, show_rate(tot_size * multiplier, duration)))

        for step in one_based_range(bench_steps):
            duration, _ = measure(lambda: sess.run(ys))
            log('step %d, took %.2fs, equivalent data rate: %s' %
                (step, duration, show_rate(tot_size * multiplier, duration)))
            
            if adapt:

                print("inside adaptation mechanism")
                if changed:
                    continue
                ret = change_strategy()
                if ret == 1:
                    changed = True


def main(_):
    args = parse_args()
    if args.method == 'HOROVOD':
        hvd_init()
    dtype = tf.float32
    sizes = _model_sizes[args.model]
    if args.fuse:
        sizes = [sum(sizes)]
    all_reduce_benchmark(sizes, dtype, args.method, args.adapt)


if __name__ == "__main__":
    main(sys.argv)
