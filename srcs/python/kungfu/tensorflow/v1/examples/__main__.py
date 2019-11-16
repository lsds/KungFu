import sys

import tensorflow as tf
from kungfu.tensorflow.ops import (all_reduce, current_cluster_size,
                                   current_rank)
from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer


def show_info_example():
    rank = current_rank()
    np = current_cluster_size()
    print('rank=%d, np=%d' % (rank, np))


def all_reduce_example():
    x = tf.Variable(tf.ones([], tf.int32))
    y = all_reduce(x)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(5):
            v = sess.run(y)
            print('step %d, result: %d' % (step, v))


def sgd_example():
    x = tf.Variable(tf.ones([], tf.float32))
    y = x * x
    lr = 0.1
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    opt = SynchronousSGDOptimizer(opt)
    train_step = opt.minimize(y)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(5):
            # v, _ = sess.run([x, train_step]) # result is not deterministic!
            sess.run(train_step)
            v = sess.run(x)
            print('step %d, result: %f' % (step, v))

            u = (1 - 2 * lr)**(step + 1)
            if abs(u - v) > 1e-6:
                msg = 'unexpected result: %f, want: %f' % (v, u)
                print(msg)
                raise RuntimeError(msg)


def main(args):
    show_info_example()
    all_reduce_example()
    sgd_example()


if __name__ == "__main__":
    main(sys.argv)
