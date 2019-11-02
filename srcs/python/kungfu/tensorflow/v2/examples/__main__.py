import sys

import tensorflow as tf
from kungfu.tensorflow.initializer import broadcast_variables
from kungfu.tensorflow.ops import current_cluster_size, current_rank
from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer


def show_info_example():
    rank = current_rank()
    np = current_cluster_size()
    print('rank=%d, np=%d' % (rank, np))


@tf.function
def training_step(x, opt, first_batch):
    with tf.GradientTape() as tape:
        y = x * x
    grads = tape.gradient(y, [x])
    opt.apply_gradients(zip(grads, [x]))

    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    if first_batch:
        broadcast_variables([x])
        broadcast_variables(opt.variables())

    return y


def test_gradient_tape():
    x = tf.Variable(tf.ones([], tf.float32))
    opt = tf.keras.optimizers.SGD(0.1)
    opt = SynchronousSGDOptimizer(opt)
    for batch in range(2):
        y = training_step(x, opt, batch == 0)


def main(args):
    show_info_example()
    test_gradient_tape()


if __name__ == "__main__":
    main(sys.argv)
