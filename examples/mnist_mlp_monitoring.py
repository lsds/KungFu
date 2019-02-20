#!/usr/bin/env python3

import argparse
import os
import time
import sys

import tensorflow as tf
from kungfu.helpers.mnist import load_datasets
from kungfu.helpers.utils import show_size

all_model_names = [
    'mnist.slp',
    'mnist.mlp',
]


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


def build_train_ops(model_name, use_kungfu):
    learning_rate = 0.1

    if model_name == 'mnist.slp':
        from kungfu.benchmarks.mnist import slp
        x, y = slp(28 * 28, 10)
    elif model_name == 'mnist.mlp':
        from kungfu.benchmarks.mnist import mlp
        x, y = mlp(28 * 28, 10, hidden_layers=[4096] * 10)
    else:
        raise RuntimeError('invalid model name: %s' % model_name)

    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    if use_kungfu:
        from kungfu.optimizers import MonitoringParallelOptimizer
        optimizer = MonitoringParallelOptimizer(optimizer)
    train_step = optimizer.minimize(loss, name='train_step')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y_, train_step, acc


def train_mnist(x, y_, train_step, acc, dataset, n_epochs=1, batch_size=5000):
    # TODO: shard by task ID
    shards = 1
    shard_id = 0

    train_data_size = 60000
    log_period = 10

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
    parser = argparse.ArgumentParser(description='KungFu mnist example.')
    parser.add_argument(
        '--use-kungfu', type=bool, default=True, help='use kungfu optimizer')
    parser.add_argument(
        '--n-epochs', type=int, default=1, help='number of epochs')
    parser.add_argument(
        '--batch-size', type=int, default=50, help='batch size')
    parser.add_argument(
        '--model-name',
        type=str,
        default='mnist.slp',
        help='model name, %s' % (' | '.join(all_model_names)))
    parser.add_argument(
        '--data-dir',
        type=str,
        default=os.path.join(os.getenv('HOME'), 'var/data/mnist'),
        help='Path to the MNIST dataset directory.')
    return parser.parse_args()


def show_info():
    g = tf.get_default_graph()
    tot_vars = 0
    tot_dim = 0
    tot_size = 0
    for v in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        dim = v.shape.num_elements()
        tot_vars += 1
        tot_dim += dim
        tot_size += dim * v.dtype.size
    print('%d vars, total dim: %d, total size: %s' % (tot_vars, tot_dim,
                                                      show_size(tot_size)))


def warmup():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


def main():
    args = parse_args()
    measure(warmup, 'warmup')
    x, y_, train_step, acc = build_train_ops(args.model_name, args.use_kungfu)
    show_info()
    mnist = measure(
        lambda: load_datasets(args.data_dir, normalize=True, one_hot=True),
        'load data')
    measure(
        lambda: train_mnist(x, y_, train_step, acc, mnist, args.n_epochs, args.
                            batch_size), 'train')


measure(main, 'main')
