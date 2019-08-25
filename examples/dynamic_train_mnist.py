#!/usr/bin/env python3
import argparse
import os
import time

import kungfu as kf
import numpy as np
import tensorflow as tf
from kungfu.helpers.mnist import load_datasets

from session import Trainer


def xentropy(y_, y):
    return -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])


def build_optimizer():
    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    from kungfu.optimizers import SyncSGDOptimizer
    optimizer = SyncSGDOptimizer(optimizer)
    return optimizer


def build_ops(images, labels, optimizer):
    from kungfu.benchmarks.layers import Dense
    y = Dense(10, act=tf.nn.softmax)(tf.reshape(images, [-1, 28 * 28]))
    loss = tf.reduce_mean(xentropy(labels, y))
    train_op = optimizer.minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return train_op, loss, accuracy


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
    args = parse_args()
    ds_train, _ds_test = create_mnist_dataset(args.data_dir)

    ds_train = ds_train.batch(args.batch_size)
    it_train = ds_train.make_one_shot_iterator()
    images, labels = it_train.get_next()

    optimizer = build_optimizer()
    train_op, loss, accuracy = build_ops(images, labels, optimizer)

    def debug(result):
        _, l, a = result
        print('loss: %f, accuracy: %f' % (l, a))

    trainer = Trainer(True)
    trainer.train(12, [train_op, loss, accuracy], debug)


main()
