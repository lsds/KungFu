"""
Usage:
    kungfu-run -q -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method CPU
    kungfu-run -q -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method NCCL
    mpirun -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method CPU
"""

import argparse
import sys
import time

import tensorflow as tf
from kungfu.tensorflow.ops import (current_cluster_size, group_all_reduce,
                                   group_nccl_all_reduce)
from kungfu.tensorflow.v1.helpers.utils import show_rate, show_size


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


def parse_args():
    p = argparse.ArgumentParser(description='Perf Benchmarks.')
    p.add_argument('--method',
                   type=str,
                   default='CPU',
                   help='CPU | NCCL | HOROVOD')
    p.add_argument('--fuse', action='store_true', default=False, help='')
    return p.parse_args()


def all_reduce_benchmark(sizes, dtype=tf.float32, method='CPU'):
    xs = [tf.Variable(tf.ones([n], dtype)) for n in sizes]
    tot_size = sum(_tensor_size(x) for x in xs)
    np = get_cluster_size(method)
    multiplier = 4 * (np - 1)
    print('all reduce %d tensors of total size: %s among %d peers, using %s' %
          (len(sizes), show_size(tot_size), np, method))

    ys = _group_all_reduce_func[method](xs)

    init = tf.global_variables_initializer()

    warmup_steps = 5
    bench_steps = 10

    with tf.Session() as sess:
        sess.run(init)
        for step in range(warmup_steps):
            sess.run(ys)
        for step in range(bench_steps):
            t0 = time.time()
            sess.run(ys)
            d = time.time() - t0
            print('step %d, took %.2fs, equivalent data rate: %s' %
                  (step, d, show_rate(tot_size * multiplier, d)))


Mi = 1 << 20


def main(_):
    args = parse_args()
    if args.method == 'HOROVOD':
        hvd_init()

    dtype = tf.float32

    single_length = 3 * Mi // dtype.size
    n = 200

    if args.fuse:
        sizes = [single_length * n]
    else:
        sizes = [single_length] * n

    all_reduce_benchmark(sizes, dtype, args.method)


if __name__ == "__main__":
    main(sys.argv)
