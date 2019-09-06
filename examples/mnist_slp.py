#!/usr/bin/env python3

import argparse
import os

import numpy as np
import tensorflow as tf

import kungfu as kf
from kungfu.helpers.mnist import load_datasets
from kungfu.helpers.utils import show_size
from kungfu.benchmarks.mnist import slp


from kungfu.internal import _get_num_peers
from kungfu.ops import all_reduce, broadcast, global_variance, group_all_reduce

from kungfu.optimizers.core import KungFuOptimizer


class VarianceBasedSyncSGDOptimizer(KungFuOptimizer):
    def __init__(self, optimizer, name=None, use_locking=False):
        super(VarianceBasedSyncSGDOptimizer,
              self).__init__(optimizer, name, use_locking)
        self._num_workers = _get_num_peers()  # FIXME: use a variable

    def apply_gradients(self, grads_and_vars, **kwargs):
        gradients, variables = list(zip(*grads_and_vars))
        squared_gradients = [tf.square(g) for g in gradients]
        summed_gradients = group_all_reduce(gradients)
        summed_squared_gradients = group_all_reduce(squared_gradients)
        reduced_grads = [g / self._num_workers for g in summed_gradients]
        reduced_square_grads = [
            g / self._num_workers for g in summed_squared_gradients
        ]
        grad_variances = [
            square_grad - tf.square(grad)
            for square_grad, grad in zip(reduced_square_grads, reduced_grads)
        ]
        variances = [
            tf.norm(grad_variance) for grad_variance in grad_variances
        ]
        self._summed_variance = tf.reduce_sum(variances)
        print(self._summed_variance)
        return self._optimizer.apply_gradients(zip(reduced_grads, variables),
                                               **kwargs)

    def distributed_initializer(self):
        ops = [tf.assign(v, broadcast(v)) for v in self.model_variables()]
        return tf.group(ops)


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

def build_optimizer(name, shards=1):
    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate / shards)
    if name == 'sync-sgd':
        from kungfu.optimizers import SyncSGDOptimizer
        return SyncSGDOptimizer(optimizer)
    if name == 'variance':
        return VarianceBasedSyncSGDOptimizer(optimizer)
    elif name == 'model-avg':
        from kungfu.optimizers import PeerModelAveragingOptimizer
        return AdaptiveModelAveragingOptimizer(optimizer)
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


def train_mnist(x, y_, train_op, optimizer, test_op, dataset, n_epochs=1,
                batch_size=5000):
    # TODO: shard by task ID
    shards = 1
    shard_id = 0

    train_data_size = 60000
    log_period = 100

    step_per_epoch = train_data_size // batch_size
    n_steps = step_per_epoch * n_epochs
    print('step_per_epoch: %d, %d steps in total' % (step_per_epoch, n_steps))

    offset = batch_size * shard_id

    kf_init_op = None
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
            offset = (offset + batch_size * shards) % train_data_size
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
