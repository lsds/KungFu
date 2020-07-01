import tensorflow as tf
from kungfu.python import run_barrier
from kungfu.tensorflow.optimizers import (PairAveragingOptimizer,
                                          SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer)
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp


def test_sync_sgd():
    x = tf.Variable(tf.ones([], tf.float32))
    y = x * x
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = SynchronousSGDOptimizer(optimizer)
    train_op = optimizer.minimize(y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(BroadcastGlobalVariablesOp())
        for _ in range(2):
            sess.run(train_op)
        # FIXME: check values


def test_sma():
    x = tf.Variable(tf.ones([], tf.float32))
    y = x * x
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = SynchronousAveragingOptimizer(optimizer)
    train_op = optimizer.minimize(y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(BroadcastGlobalVariablesOp())
        for _ in range(2):
            sess.run(train_op)
        # FIXME: check values


def test_pair_averaging():
    x = tf.Variable(tf.ones([], tf.float32))
    y = x * x
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = PairAveragingOptimizer(optimizer)
    train_op = optimizer.minimize(y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(BroadcastGlobalVariablesOp())
        for _ in range(2):
            sess.run(train_op)
        # FIXME: check values
    run_barrier()


test_sync_sgd()
test_sma()
test_pair_averaging()
