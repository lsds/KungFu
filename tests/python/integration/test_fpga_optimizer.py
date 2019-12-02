import tensorflow as tf
from kungfu import run_barrier
from kungfu.tensorflow.optimizers.sync_sgd_fpga import FPGASynchronousSGDOptimizer
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp


def test_fpga_sync_sgd():
    x = tf.Variable(tf.ones([], tf.float32))
    y = x * x
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = FPGASynchronousSGDOptimizer(optimizer)
    train_op = optimizer.minimize(y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(BroadcastGlobalVariablesOp())
        for _ in range(2):
            sess.run(train_op)
        # FIXME: check values


test_fpga_sync_sgd()
