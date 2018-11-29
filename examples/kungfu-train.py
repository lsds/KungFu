#!/usr/bin/env python3

import argparse
import os
import time
import sys

import tensorflow as tf
from kungfu.helpers.mnist import load_datasets

all_model_names = ['mnist.slp', 'mnist.mlp']


def measure(f, name=None):
    if not name:
        name = f.__name__
    t0 = time.time()
    result = f()
    duration = time.time() - t0
    line = '%s took %fs' % (name, duration)
    print(line)
    with open('profile.log', 'a') as f:
        f.write(line + '\n')
    return result


def build_train_ops(model_name, use_async_sgd):
    learning_rate = 0.1

    if model_name == 'mnist.slp':
        from kungfu.benchmarks.mnist import slp
        x, y = slp(28 * 28, 10)
    elif model_name == 'mnist.mlp':
        from kungfu.benchmarks.mnist import mlp
        x, y = mlp(28 * 28, 10)
    else:
        raise RuntimeError('invalid model name: %s' % model_name)

    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    optmizer = tf.train.GradientDescentOptimizer(learning_rate)
    if use_async_sgd:
        from kungfu import AsyncSGDOptimizer
        optmizer = AsyncSGDOptimizer(optmizer)
    train_step = optmizer.minimize(loss, name='train_step')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y_, train_step, acc


def train(x, y_, train_step, acc, dataset):
    shards = 1
    shard_id = 0

    train_data_size = 60000
    batch_size = 5000
    log_period = 1
    n_epochs = 1

    step_per_epoch = train_data_size // batch_size
    n_steps = step_per_epoch * n_epochs
    print('step_per_epoch: %d, %d steps in total' % (step_per_epoch, n_steps))

    offset = batch_size * shard_id

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(1, n_steps + 1):
            print('step: %d' % step)
            xs = dataset.train.images[offset:offset + batch_size, :]
            y_s = dataset.train.labels[offset:offset + batch_size]
            offset = (offset + batch_size * shards) % train_data_size
            sess.run(train_step, {
                x: xs.reshape(batch_size, 28 * 28),
                y_: y_s,
            })
            if step % log_period == 0:
                result = sess.run(
                    acc, {
                        x: dataset.test.images.reshape(10000, 28 * 28),
                        y_: dataset.test.labels
                    })
                print('validation accuracy: %f' % result)

        result = sess.run(
            acc, {
                x: dataset.test.images.reshape(10000, 28 * 28),
                y_: dataset.test.labels
            })
        print('test accuracy: %f' % result)


def parse_args():
    parser = argparse.ArgumentParser(description='kungfu-example')
    parser.add_argument(
        '--use-async-sgd', type=bool, default=True, help='use async SGD')

    parser.add_argument(
        '--model-name',
        type=str,
        default='mnist.slp',
        help='model name, %s' % (' | '.join(all_model_names)))
    return parser.parse_args()


def main():
    args = parse_args()
    x, y_, train_step, acc = build_train_ops(args.model_name,
                                             args.use_async_sgd)
    data_dir = os.path.join(os.getenv('HOME'), 'var/data/mnist')
    mnist = measure(
        lambda: load_datasets(data_dir, normalize=True, one_hot=True),
        'load data')
    measure(lambda: train(x, y_, train_step, acc, mnist), 'train')


measure(main, 'main')
