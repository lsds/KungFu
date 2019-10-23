import sys
import time

import tensorflow as tf
from kungfu.helpers.utils import show_rate, show_size
from kungfu.ops import _tensor_size, current_cluster_size, group_all_reduce


def all_reduce_benchmark(sizes, dtype=tf.float32):
    xs = [tf.Variable(tf.ones([n], dtype)) for n in sizes]
    tot_size = sum(_tensor_size(x) for x in xs)
    np = current_cluster_size()
    multiplier = 4 * (np - 1)
    print('all reduce total size: %s among %d peers' %
          (show_size(tot_size), np))
    ys = group_all_reduce(xs)
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
            rate = 0
            print('step %d, took %.2fs, equivalent data rate: %s' %
                  (step, d, show_rate(tot_size * multiplier, d)))


def main(args):
    all_reduce_benchmark([1 << 10])
    all_reduce_benchmark([1 << 20])
    all_reduce_benchmark([25 << 20])


if __name__ == "__main__":
    main(sys.argv)
