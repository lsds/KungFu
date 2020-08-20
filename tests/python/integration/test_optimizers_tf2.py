import tensorflow as tf
from kungfu.python import run_barrier
from kungfu.tensorflow.optimizers import (PairAveragingOptimizer,
                                          SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer)
from kungfu.tensorflow.initializer import broadcast_variables


def _training_step(x, opt, first_batch):
    with tf.GradientTape() as tape:
        y = x * x
    grads = tape.gradient(y, [x])
    opt.apply_gradients(zip(grads, [x]))

    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    if first_batch:
        broadcast_variables([x])
        broadcast_variables(opt.variables())

    return y


def test_sync_sgd():
    x = tf.Variable(tf.ones([], tf.float32))
    opt = tf.keras.optimizers.SGD(0.1)
    opt = SynchronousSGDOptimizer(opt)

    @tf.function
    def training_step(x, opt, first_batch):
        _training_step(x, opt, first_batch)

    for batch in range(5):
        y = training_step(x, opt, batch == 0)
        # FIXME: check values


def test_sma():
    x = tf.Variable(tf.ones([], tf.float32))
    opt = tf.keras.optimizers.SGD(0.1)
    opt = SynchronousAveragingOptimizer(opt)

    @tf.function
    def training_step(x, opt, first_batch):
        _training_step(x, opt, first_batch)

    for batch in range(5):
        y = training_step(x, opt, batch == 0)
        # FIXME: check values


def test_pair_averaging():
    x = tf.Variable(tf.ones([], tf.float32))
    opt = tf.keras.optimizers.SGD(0.1)
    opt = PairAveragingOptimizer(opt)

    @tf.function
    def training_step(x, opt, first_batch):
        _training_step(x, opt, first_batch)

    for batch in range(5):
        y = training_step(x, opt, batch == 0)
        # FIXME: check values
    run_barrier()


test_sync_sgd()
test_sma()
test_pair_averaging()
