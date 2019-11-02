#!/usr/bin/env python3

import argparse
import os

import tensorflow as tf
from kungfu.tensorflow.v1.helpers.mnist import load_datasets


def new_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def new_weight(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def dense(x, logits, act):
    _n, m = x.shape
    input_size = int(m)
    w = new_weight((input_size, logits))
    y = tf.matmul(x, w)
    b = new_bias((logits, ))
    y = tf.add(y, b)
    if act:
        y = act(y)
    return y


def fake_get_shard_info(use_kungfu):
    if use_kungfu:
        from kungfu.tensorflow.ops import current_cluster_size, current_rank
        return current_rank(), current_cluster_size()
    return 0, 1


def slp(input_size, logits):
    x = tf.placeholder(tf.float32, [None, input_size])
    y = dense(x, logits, act=tf.nn.softmax)
    return x, y


def xentropy(y_, y):
    return -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])


def build_optimizer(shards, use_kungfu=True):
    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    if use_kungfu:
        from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
        optimizer = SynchronousSGDOptimizer(optimizer)
    return optimizer


def build_ops(optimizer):
    x, y = slp(28 * 28, 10)
    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(xentropy(y_, y))
    train_op = optimizer.minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    test_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_, train_op, test_op)


def divide(a, b):
    return a // b, a % b


class MnistTrainer(object):
    def __init__(self, optimizer, batch_size, n_epochs, data_dir, shards):
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._ops = build_ops(optimizer)
        self._dataset = load_datasets(data_dir, normalize=True, one_hot=True)
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def run(self, shard_id, shards):
        train_data_size = 60000
        log_period = 100

        n_steps, rem = divide(train_data_size, self._batch_size * shards)
        if rem:
            raise RuntimeError('%d is not a multiple of batch_size * shards' %
                               (train_data_size))
        print('%d steps per epoch' % (n_steps))

        for epoch in range(1, 1 + self._n_epochs):
            print('epoch: %d' % epoch)
            offset = self._batch_size * shard_id
            for step in range(1, 1 + n_steps):
                self.train(offset)
                offset += self._batch_size * shards
                if step % log_period == 0:
                    result = self.validate()
                    print('step %d, validate accuracy: %f' % (step, result))

        result = self.test()
        print('test accuracy: %f' % result)
        return result

    def train(self, offset):
        xs = self._dataset.train.images[offset:offset + self._batch_size, :]
        y_s = self._dataset.train.labels[offset:offset + self._batch_size]
        (x, y_, train_op, _test_op) = self._ops
        self._sess.run(train_op, {
            x: xs.reshape(self._batch_size, 28 * 28),
            y_: y_s,
        })

    def validate(self):
        return self.test()

    def test(self):
        (x, y_, _train_op, test_op) = self._ops
        result = self._sess.run(
            test_op, {
                x: self._dataset.test.images.reshape(10000, 28 * 28),
                y_: self._dataset.test.labels
            })
        return result


def parse_args():
    parser = argparse.ArgumentParser(description='KungFu mnist example.')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=1,
                        help='number of epochs')
    parser.add_argument('--batch-size',
                        type=int,
                        default=500,
                        help='batch size')
    parser.add_argument('--data-dir',
                        type=str,
                        default=os.path.join(os.getenv('HOME'),
                                             'var/data/mnist'),
                        help='Path to the MNIST dataset directory.')
    parser.add_argument('--no-kungfu',
                        type=bool,
                        default=False,
                        help='disable kungfu')
    return parser.parse_args()


def float_eq(x, y):
    return abs(x - y) < 1e-6


def check_result(result, batch_size, cluster_size, n_epochs):
    expected_results = {
        (50, 2): 0.913700,
        (500, 2): 0.888400,
        (600, 2): 0.884500,
        (6000, 2): 0.819300,
    }

    expect = expected_results[(batch_size * cluster_size, n_epochs)]
    if not float_eq(expect, result):
        raise RuntimeError(
            'unexpected result: %f, (%d, %d, %d) should be %f' %
            (result, batch_size, cluster_size, n_epochs, expect))


def main():
    args = parse_args()
    use_kungfu = not args.no_kungfu
    rank, cluster_size = fake_get_shard_info(use_kungfu)
    optimizer = build_optimizer(cluster_size, use_kungfu)
    trainer = MnistTrainer(optimizer, args.batch_size, args.n_epochs,
                           args.data_dir, cluster_size)
    result = trainer.run(rank, cluster_size)
    check_result(result, args.batch_size, cluster_size, args.n_epochs)


main()
