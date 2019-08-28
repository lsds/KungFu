#!/usr/bin/env python3
import argparse
import os
import time

import kungfu as kf
import numpy as np
import tensorflow as tf
from kungfu.helpers.cifar import Cifar10Loader, Cifar100Loader

#
from session import Trainer
from dataset import DynamicDatasetAdaptor


def xentropy(y_, y):
    return -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])


def build_optimizer(local_batch_size):
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # from kungfu.optimizers import SyncSGDOptimizer
    # optimizer = SyncSGDOptimizer(optimizer)
    from gns_optimizer import GradientNoiseScaleAdaptiveOptimizer
    optimizer = GradientNoiseScaleAdaptiveOptimizer(optimizer,
                                                    local_batch_size)
    return optimizer


def build_model(x, model='slp', logits=10):
    if model == 'slp':
        from kungfu.benchmarks.layers import Dense
        y = Dense(logits, act=tf.nn.softmax)(tf.reshape(x, [-1, 32 * 32 * 3]))
    elif model == 'cnn':
        from kungfu.benchmarks.layers import Conv, Dense, Pool, seq_apply
        layers = [
            Conv([5, 5], 32, act=tf.nn.relu),
            Pool(),
            Conv([3, 3], 64, act=tf.nn.relu),
            Pool(),
            Conv([3, 3], 128, act=tf.nn.relu),
            Pool(),
            Conv([3, 3], 256, act=tf.nn.relu),
            Pool(),
            Dense(logits, act=tf.nn.softmax),
        ]
        y = seq_apply(layers, x)
    elif model == 'ResNet50':
        from tensorflow.keras import applications
        raise RuntimeError('TODO %s' % model)
        # return getattr(applications, model)(weights=None)(x, training=True)
    else:
        raise RuntimeError('invalid model %s' % model)
    return y


def build_ops(output, labels, optimizer):
    loss = tf.reduce_mean(xentropy(labels, output))
    train_op = optimizer.minimize(loss)
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return train_op, loss, accuracy


def create_labeled_dataset(data, repeat=False):
    images = tf.data.Dataset.from_tensor_slices(data.images)
    labels = tf.data.Dataset.from_tensor_slices(data.labels)
    if repeat:
        images = images.repeat()
        labels = labels.repeat()
    return tf.data.Dataset.zip((images, labels))


def create_cifar_dataset(model, data_dir, repeat):
    loader_class = {
        'cifar10': Cifar10Loader,
        'cifar100': Cifar100Loader
    }[model]
    loader = loader_class(data_dir, normalize=True, one_hot=True)
    ds = loader.load_datasets()
    ds_train = create_labeled_dataset(ds.train, repeat=repeat)
    ds_test = create_labeled_dataset(ds.test)
    return ds_train, ds_test


def parse_args():
    parser = argparse.ArgumentParser(description='KungFu cifar10 example.')
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        help='cifar10 | cifar100')
    parser.add_argument('--model',
                        type=str,
                        default='slp',
                        help='model name : slp | cnn')
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
                                             'var/data/cifar'),
                        help='Path to the CIFAR dataset directory.')
    return parser.parse_args()


def main():
    args = parse_args()
    logits = {'cifar10': 10, 'cifar100': 100}[args.dataset]

    ds_train, _ds_test = create_cifar_dataset(args.dataset, args.data_dir,
                                              True)

    adaptor = DynamicDatasetAdaptor(batch_size=args.batch_size)
    init_train, it_train = adaptor(ds_train)
    images, labels = it_train

    output = build_model(images, args.model, logits)
    optimizer = build_optimizer(args.batch_size)
    train_op, loss, accuracy = build_ops(output, labels, optimizer)

    def debug(result):
        (_, l, a, bs, bs2) = result
        print('loss: %f, accuracy: %f, bs: %f, gbs: %f' % (l, a, bs, bs2))

    trainer = Trainer()
    trainer.train(args.max_step, [
        train_op,
        loss,
        accuracy,
        optimizer._predicated_local_batch_size,
        optimizer._predicated_global_batch_size,
    ], debug, [init_train])


main()
