"""
Usage:
    kungfu_run 4 python3 -m kungfu.tensorflow.v1.benchmarks --device NCCL
    kungfu_run 4 python3 -m kungfu.tensorflow.v1.benchmarks --device CPU
"""

import argparse
import sys
import time

from kungfu.tensorflow.ops import (current_cluster_size, group_all_reduce,
                                   group_nccl_all_reduce)
from kungfu.tensorflow.v1.helpers.utils import show_rate, show_size

import tensorflow as tf


def _tensor_size(t):
    return t.shape.num_elements() * t.dtype.size


def parse_args():
    p = argparse.ArgumentParser(description='Perf Benchmarks.')
    p.add_argument('--device', type=str, default='CPU', help='CPU | NCCL')
    return p.parse_args()


_group_all_reduce_func = {
    'CPU': group_all_reduce,
    'NCCL': group_nccl_all_reduce,
}


def all_reduce_benchmark(sizes, dtype=tf.float32, device='CPU'):
    xs = [tf.Variable(tf.ones([n], dtype)) for n in sizes]
    tot_size = sum(_tensor_size(x) for x in xs)
    np = current_cluster_size()
    multiplier = 4 * (np - 1)
    print('all reduce %d tensors of total size: %s among %d peers, using %s' %
          (len(sizes), show_size(tot_size), np, device))
    ys = _group_all_reduce_func[device](xs)
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

    dtype = tf.float32

    single_length = 3 * Mi // dtype.size
    n = 200

    all_reduce_benchmark([single_length] * n, dtype, args.device)
    all_reduce_benchmark([single_length * n], dtype, args.device)


if __name__ == "__main__":
    main(sys.argv)
