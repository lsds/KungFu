#!/usr/bin/env python3
# This example shows how a MNIST Single Layer Perception Model training program
# can adopt various distributed synchronization strategies using KungFu.
#
# In principle, KungFu requires users to make the following changes:
# 1. KungFu provides distributed optimizers that can wrap the original optimizer.
# The distributed optimizer defines how local gradients and model weights are synchronized.
# 2. (Optional) In a distributed training setting, the training dataset is often partitioned.
# 3. (Optional) Scaling the learning rate of your local optimizer

import argparse
import os

import kungfu as kf
import numpy as np
import tensorflow as tf
from kungfu import current_cluster_size, current_rank
from kungfu.tensorflow.v1.helpers.mnist import load_datasets


def save_vars(sess, variables, filename):
    values = sess.run(variables)
    npz = dict((var.name, val) for var, val in zip(variables, values))
    np.savez(filename, **npz)


def save_all(sess, prefix):
    g = tf.get_default_graph()
    filename = '%s-%d.npz' % (prefix, os.getpid())
    save_vars(sess, g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), filename)


def load_mnist(data_dir):
    dataset = dict()
    dataset['training_set'] = dict()
    dataset['validation_set'] = dict()
    dataset['test_set'] = dict()

    mnist = load_datasets(data_dir, normalize=True, one_hot=True)

    # reshape the inputs
    set_size = mnist.train.images.shape[0]
    test_set_size = mnist.test.images.shape[0]
    dataset['training_set']['x'] = mnist.train.images.reshape(
        set_size, 28 * 28)
    dataset['test_set']['x'] = mnist.test.images.reshape(
        test_set_size, 28 * 28)

    # split into training and validation set
    validation_set_size = set_size // 6
    training_set_size = set_size - validation_set_size
    dataset['validation_set']['x'] = dataset['training_set']['x'][
        0:validation_set_size]
    dataset['validation_set']['y'] = mnist.train.labels[0:validation_set_size]
    dataset['training_set']['x'] = dataset['training_set']['x'][
        validation_set_size:set_size]
    dataset['training_set']['y'] = mnist.train.labels[validation_set_size:
                                                      set_size]
    dataset['test_set']['y'] = mnist.test.labels

    return dataset


# instantiate the optimizer
def build_optimizer(name, batch_size):
    learning_rate = 0.1

    # Scale learning rate according to the level of data parallelism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate *
                                                  current_cluster_size())

    # KungFu: Wrap the TensorFlow optimizer with KungFu distributed optimizers.
    if name == 'sync-sgd':
        from kungfu.tensorflow.v1.optimizers import SynchronousSGDOptimizer
        return SynchronousSGDOptimizer(optimizer)
    elif name == 'async-sgd':
        from kungfu.tensorflow.v1.optimizers import PairAveragingOptimizer
        return PairAveragingOptimizer(optimizer)
    elif name == 'sma':
        from kungfu.tensorflow.v1.optimizers import SynchronousAveragingOptimizer
        return SynchronousAveragingOptimizer(optimizer)
    elif name == 'noise-scale':
        from kungfu.tensorflow.v1.optimizers import SyncSGDWithGradNoiseScaleOptimizer
        return SyncSGDWithGradNoiseScaleOptimizer(optimizer,
                                                  device_batch_size=batch_size)
    else:
        raise RuntimeError('unknown optimizer: %s' % name)


def build_model(optimizer):
    input_size = 28 * 28
    num_classes = 10

    # create a placeholder for the input
    x = tf.placeholder(tf.float32, [None, input_size])
    # add a dense layer
    y = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(x)

    # create a placeholder for the true labels
    y_ = tf.placeholder(tf.float32, [None, 10])
    # use cross entropy for the loss
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
    loss = tf.reduce_mean(cross_entropy)
    # minimise the loss
    train_op = optimizer.minimize(loss)

    # calculate the number of correctly classified datapoints
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    test_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return (x, y_, train_op, test_op)


# evaluate accuracy of the learned model
def test_mnist(sess, x, y_, test_op, test_set):
    result = sess.run(test_op, {
        x: test_set['x'],
        y_: test_set['y'],
    })

    return result


# train the model on the mnist training set
def train_mnist(sess,
                x,
                y_,
                train_op,
                test_op,
                optimizer,
                dataset,
                n_epochs=1,
                batch_size=5000):

    log_period = 100

    # get the cluster size
    n_shards = current_cluster_size()
    # get the cluster rank of the node
    shard_id = current_rank()

    # calculate number of datapoints per node
    training_set_size = dataset['training_set']['x'].shape[0]
    shard_size = training_set_size // n_shards
    step_per_epoch = shard_size // batch_size
    n_steps = step_per_epoch * n_epochs
    print('step_per_epoch: %d, %d steps in total' % (step_per_epoch, n_steps))

    # KungFu: Each replica is responsible for a data shard.
    offset = batch_size * shard_id

    sess.run(tf.global_variables_initializer())

    # KungFu: broadcast the global variable
    from kungfu.tensorflow.v1.initializer import BroadcastGlobalVariablesOp
    sess.run(BroadcastGlobalVariablesOp())

    print('training')
    # train the model with all batches allocated to the node
    for step in range(n_steps):
        xs = dataset['training_set']['x'][offset:offset + batch_size]
        y_s = dataset['training_set']['y'][offset:offset + batch_size]
        offset = (offset + batch_size * n_shards) % training_set_size
        sess.run(train_op, {
            x: xs,
            y_: y_s,
        })
        # log the validation accuracy
        if step % log_period == 0:
            training_acc_dataset = dict()
            training_acc_dataset['x'] = xs
            training_acc_dataset['y'] = y_s
            result = test_mnist(sess, x, y_, test_op, training_acc_dataset)
            print('training accuracy: %f' % result)
            result = test_mnist(sess, x, y_, test_op,
                                dataset['validation_set'])
            print('validation accuracy: %f' % result)


# parse arguments from the command line
def parse_args():
    parser = argparse.ArgumentParser(description='KungFu mnist example.')
    parser.add_argument('--kf-optimizer',
                        type=str,
                        default='sync-sgd',
                        help='kungfu optimizer')
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
                        default='mnist',
                        help='Path to the MNIST dataset directory.')
    return parser.parse_args()


def main():
    args = parse_args()
    optimizer = build_optimizer(name=args.kf_optimizer,
                                batch_size=args.batch_size)
    x, y_, train_op, test_op = build_model(optimizer)
    mnist = load_mnist(args.data_dir)

    with tf.Session() as sess:
        train_mnist(sess, x, y_, train_op, test_op, optimizer, mnist,
                    args.n_epochs, args.batch_size)
        result = test_mnist(sess, x, y_, test_op, mnist['test_set'])
        print('test accuracy: %f' % result)
        # save_all(sess, 'final')


main()
