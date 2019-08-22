#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import tensorflow as tf

import kungfu as kf
from kungfu.helpers.mnist import load_datasets
from kungfu.benchmarks.mnist import slp
from kungfu.ops import get_init_version, propose_update, broadcast, barrier, save_variable, all_reduce
from session import kungfu_train


def xentropy(y_, y):
    return -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])


def build_optimizer():
    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    from kungfu.optimizers import SyncSGDOptimizer
    optimizer = SyncSGDOptimizer(optimizer)
    return optimizer


def build_ops(optimizer):
    x, y = slp(28 * 28, 10)
    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(xentropy(y_, y))
    train_op = optimizer.minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    test_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_, train_op, test_op)


class StopWatch():
    def __init__(self):
        self._last = time.time()

    def __call__(self):
        t = time.time()
        d = t - self._last
        self._last = t
        return d


def train_mnist(x, y_, train_op, test_op, ds_train, ds_test, batch_size=5000):
    ds_train = ds_train.batch(batch_size)
    it_train = ds_train.make_one_shot_iterator()
    get_next_train = it_train.get_next()

    def train_step(sess):
        xs, y_s = sess.run(get_next_train)
        print('train one step with xs :: %s, y_s :: %s' %
              (xs.shape, y_s.shape))
        sess.run(train_op, {
            x: xs.reshape(-1, 28 * 28),
            y_: y_s,
        })

    test_batch_size = 10000
    ds_test = ds_test.batch(test_batch_size).repeat()
    it_test = ds_test.make_one_shot_iterator()
    get_next_test = it_test.get_next()

    def test_step(sess):
        xs, y_s = sess.run(get_next_test)
        result = sess.run(test_op, {
            x: xs.reshape(-1, 28 * 28),
            y_: y_s,
        })
        return result

    kungfu_train(12, train_step)


def create_labeled_dataset(data):
    images = tf.data.Dataset.from_tensor_slices(data.images)
    labels = tf.data.Dataset.from_tensor_slices(data.labels)
    return tf.data.Dataset.zip((images, labels))


def create_mnist_dataset(data_dir):
    mnist = load_datasets(data_dir, normalize=True, one_hot=True)
    ds_train = create_labeled_dataset(mnist.train)
    ds_test = create_labeled_dataset(mnist.test)
    return ds_train, ds_test


def parse_args():
    parser = argparse.ArgumentParser(description='KungFu mnist example.')
    parser.add_argument('--use-dynamic', type=bool, default=True, help='')
    parser.add_argument('--batch-size',
                        type=int,
                        default=500,
                        help='batch size')
    parser.add_argument('--data-dir',
                        type=str,
                        default=os.path.join(os.getenv('HOME'),
                                             'var/data/mnist'),
                        help='Path to the MNIST dataset directory.')
    return parser.parse_args()


def main():
    one = tf.Variable(tf.ones([], dtype=tf.int32))
    np = all_reduce(one)

    def train_step(sess):
        v = sess.run(np)
        print('finished train step: np=%d' % (v))

    kungfu_train(12, train_step)

    # args = parse_args()
    # ds_train, ds_test = create_mnist_dataset(args.data_dir)
    # optimizer = build_optimizer()
    # x, y_, train_op, test_op = build_ops(optimizer)
    # train_mnist(x, y_, train_op, test_op, ds_train, ds_test, args.batch_size)


main()
