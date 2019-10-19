from kungfu.optimizers import SyncSGDOptimizer
import tensorflow as tf


def test_sync_sgd():
    x = tf.Variable(tf.ones([], tf.float32))
    y = x * x
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = SyncSGDOptimizer(optimizer)
    train_op = optimizer.minimize(y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            sess.run(train_op)
        # FIXME: check values


test_sync_sgd()

# TODO: more tests
