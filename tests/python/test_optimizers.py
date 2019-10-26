from kungfu.tensorflow.v1.optimizers import SynchronousSGDOptimizer
from kungfu.tensorflow.v1.optimizers import PairAveragingOptimizer
import tensorflow as tf


def test_sync_sgd():
    x = tf.Variable(tf.ones([], tf.float32))
    y = x * x
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = SynchronousSGDOptimizer(optimizer)
    train_op = optimizer.minimize(y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            sess.run(train_op)
        # FIXME: check values


test_sync_sgd()


def test_pair_averaging():
    x = tf.Variable(tf.ones([], tf.float32))
    y = x * x
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = PairAveragingOptimizer(optimizer)
    train_op = optimizer.minimize(y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            sess.run(train_op)
        # FIXME: check values


test_pair_averaging()
