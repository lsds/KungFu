#!/usr/bin/env python3

import tensorflow as tf
from kungfu.cmd import launch_multiprocess
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


def worker(rank):
    from kungfu.python import current_cluster_size, current_rank
    from kungfu.tensorflow.ops import all_reduce
    print('rank=%d' % (rank))
    print('kungfu rank: %d, size %d' %
          (current_rank(), current_cluster_size()))
    x = tf.Variable(tf.ones(shape=(), dtype=tf.int32))
    y = all_reduce(x * rank)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        v = sess.run(y)
        print('v=%s' % (v))


def main():
    np = 4
    launch_multiprocess(worker, np)


main()
