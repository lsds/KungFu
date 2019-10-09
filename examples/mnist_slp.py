#!/usr/bin/env python3
# This example shows how a MNIST Single Layer Perception Model training program
# can adopt various distributed synchronization strategies using KungFu.
#
# In principle, KungFu requires users to make three changes:
# 1. KungFu provides distributed optimizers that can wrap the original optimizer.
# The distributed optimizer defines how local gradients and model weights are synchronized.
# 2. KungFu provides distributed variable initializers that defines how model weights are
# initialized on distributed devices.
# 3. (Optional) In a distributed training setting, the training dataset is often partitioned.

import argparse
import os

import kungfu as kf
import numpy as np
import tensorflow as tf
from kungfu.benchmarks.mnist import slp
from kungfu.helpers.mnist import load_datasets
from kungfu.helpers.utils import show_size
from kungfu.ops import current_cluster_size, current_rank


def save_vars(sess, variables, filename):
    values = sess.run(variables)
    npz = dict((var.name, val) for var, val in zip(variables, values))
    np.savez(filename, **npz)


def save_all(sess, prefix):
    g = tf.get_default_graph()
    filename = '%s-%d.npz' % (prefix, os.getpid())
    save_vars(sess, g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), filename)


def xentropy(y_, y):
    return -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])


def build_optimizer(name, n_shards=1):
    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate / n_shards)

    # KUNGFU: Wrap the TensorFlow optimizer with KungFu distributed optimizers.
    if name == 'sync-sgd':
        from kungfu.optimizers import SyncSGDOptimizer
        return SyncSGDOptimizer(optimizer)
    if name == 'variance':
        from kungfu.optimizers import SyncSGDWithGradVarianceOptimizer
        return SyncSGDWithGradVarianceOptimizer(optimizer, monitor_interval=10)
    elif name == 'model-avg':
        from kungfu.optimizers import PeerModelAveragingOptimizer
        return PeerModelAveragingOptimizer(optimizer)
    else:
        raise RuntimeError('unknow optimizer: %s' % name)


def build_ops(optimizer):
    x, y = slp(28 * 28, 10)
    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(xentropy(y_, y))
    train_op = optimizer.minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    test_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_, train_op, test_op)


def test_mnist(sess, x, y_, test_op, dataset):
    result = sess.run(test_op, {
        x: dataset.images.reshape(10000, 28 * 28),
        y_: dataset.labels,
    })
    return result


def train_mnist(x,
                y_,
                train_op,
                optimizer,
                test_op,
                dataset,
                n_epochs=1,
                batch_size=5000):
    n_shards = current_cluster_size()
    shard_id = current_rank()

    train_data_size = 60000
    log_period = 100

    shard_size = train_data_size // n_shards
    step_per_epoch = shard_size // batch_size
    n_steps = step_per_epoch * n_epochs
    print('step_per_epoch: %d, %d steps in total' % (step_per_epoch, n_steps))

    # KUNGFU: Each replica is responsible for a data shard.
    offset = batch_size * shard_id

    kf_init_op = None

    # KUNGFU: KungFu initilizer defines how model weights are initilised on distributed devices
    if hasattr(optimizer, 'distributed_initializer'):
        kf_init_op = optimizer.distributed_initializer()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if kf_init_op:
            sess.run(kf_init_op)

        print('training')
        for step in range(1, n_steps + 1):
            xs = dataset.train.images[offset:offset + batch_size, :]
            y_s = dataset.train.labels[offset:offset + batch_size]
            offset = (offset + batch_size * n_shards) % train_data_size
            sess.run(train_op, {
                x: xs.reshape(batch_size, 28 * 28),
                y_: y_s,
            })
            if step % log_period == 0:
                # FIXME: don't use test data for validation
                result = test_mnist(sess, x, y_, test_op, dataset.test)
                print('validation accuracy: %f' % result)

        result = test_mnist(sess, x, y_, test_op, dataset.test)
        print('test accuracy: %f' % result)
        # save_all(sess, 'final')


def parse_args():
    parser = argparse.ArgumentParser(description='KungFu mnist example.')
    parser.add_argument('--optimizer', type=str, default='sync-sgd', help='')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=1,
                        help='number of epochs')
    parser.add_argument('--batch-size',
                        type=int,
                        default=50,
                        help='batch size')
    parser.add_argument('--data-dir',
                        type=str,
                        default=os.path.join(os.getenv('HOME'),
                                             'var/data/mnist'),
                        help='Path to the MNIST dataset directory.')
    return parser.parse_args()


def main():
    args = parse_args()
    optimizer = build_optimizer(args.optimizer)
    x, y_, train_op, test_op = build_ops(optimizer)

    mnist = load_datasets(args.data_dir, normalize=True, one_hot=True)
    train_mnist(x, y_, train_op, optimizer, test_op, mnist, args.n_epochs,
                args.batch_size)


main()
