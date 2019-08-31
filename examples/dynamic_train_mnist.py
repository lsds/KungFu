#!/usr/bin/env python3
import argparse
import os
import time

import kungfu as kf
import numpy as np
import tensorflow as tf
from kungfu.helpers.mnist import load_datasets

#
from session import Trainer
from dataset import DynamicDatasetAdaptor
from common import get_rank, Reporter


def xentropy(y_, y):
    return -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])


def build_optimizer(local_batch_size):
    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # from kungfu.optimizers import SyncSGDOptimizer
    # optimizer = SyncSGDOptimizer(optimizer)
    from gns_optimizer import GradientNoiseScaleAdaptiveOptimizer
    optimizer = GradientNoiseScaleAdaptiveOptimizer(optimizer,
                                                    local_batch_size)
    return optimizer


def build_ops(images, labels, optimizer):
    from kungfu.benchmarks.layers import Dense
    y = Dense(10, act=tf.nn.softmax)(tf.reshape(images, [-1, 28 * 28]))
    loss = tf.reduce_mean(xentropy(labels, y))
    train_op = optimizer.minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return train_op, loss, accuracy


def create_labeled_dataset(data, repeat=False):
    images = tf.data.Dataset.from_tensor_slices(data.images)
    labels = tf.data.Dataset.from_tensor_slices(data.labels)
    if repeat:
        images = images.repeat()
        labels = labels.repeat()
    return tf.data.Dataset.zip((images, labels))


def create_mnist_dataset(data_dir, repeat):
    ds = load_datasets(data_dir, normalize=True, one_hot=True)
    ds_train = create_labeled_dataset(ds.train, repeat=repeat)
    ds_test = create_labeled_dataset(ds.test)
    return ds_train, ds_test


def parse_args():
    parser = argparse.ArgumentParser(description='KungFu mnist example.')
    parser.add_argument('--adaptive',
                        type=bool,
                        default=False,
                        help='adaptive')
    parser.add_argument('--max-step',
                        type=int,
                        default=1200,
                        help='max training steps')
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
    init_rank = get_rank()

    ds_train, _ds_test = create_mnist_dataset(args.data_dir, True)

    adaptor = DynamicDatasetAdaptor(batch_size=args.batch_size)
    init_train, it_train = adaptor(ds_train)
    images, labels = it_train

    optimizer = build_optimizer(args.batch_size)
    train_op, loss, accuracy = build_ops(images, labels, optimizer)

    report = Reporter('%f %f %f\n')

    def debug(result):
        (_, l, a, bs, gbs) = result
        print('loss: %f, accuracy: %f, bs: %f, gbs: %f' % (l, a, bs, gbs))
        report([l, a, gbs])

    trainer = Trainer(args.adaptive)
    trainer.train(args.max_step, [
        train_op,
        loss,
        accuracy,
        optimizer._predicated_local_batch_size,
        optimizer._predicated_global_batch_size,
    ], debug, [init_train])

    if init_rank == 0:
        report.save('result-%02d.txt' % init_rank)


main()
