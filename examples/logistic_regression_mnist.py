'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

#!/usr/bin/env python3

import argparse
import os
import time
import timeit
import sys

import datetime
import numpy as np
import tensorflow as tf
from kungfu.helpers.mnist import load_datasets
from kungfu.helpers.utils import show_size

import tensorflow.examples.tutorials.mnist.input_data as input_data
import kungfu as kf

# %%
# get the classic mnist dataset
# one-hot means a sparse vector for every observation where only
# the class label is 1, and every other class is 0.
# more info here:
# https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/download/index.html#dataset-object
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)



# TODO: add to kungfu optimizer; use model size in bits x64
def get_number_of_trainable_parameters():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

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

def build_train_ops(use_kungfu, kungfu_strategy, ako_partitions, staleness, kickin_time):
    # Parameters
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_step = 1

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

    # Set model weights
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

    # Minimize error using cross entropy
    loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

    optmizer = tf.train.GradientDescentOptimizer(learning_rate)
    if use_kungfu:
        optmizer = kf.SyncSGDOptimizer(optmizer, strategy=kungfu_strategy,
                                      ako_partitions=ako_partitions,
                                      staleness=staleness,
                                      kickin_time=kickin_time)

    train_step = optmizer.minimize(loss, name='train_step')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y, train_step, acc


def train_mnist(x, y, train_step, acc, n_epochs, batch_size, val_accuracy_target):
    n_epochs = 25
    batch_size = 50

    def  evaluate_test_set_accuracy(acc):
        test_acc = sess.run(acc,
                    feed_dict={
                        x: mnist.test.images,
                        y: mnist.test.labels
        })
        print('test accuracy: %f' % test_acc)

    time_start = time.time()
    reached_target_accuracy = False
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(kf.distributed_variables_initializer())

        for epoch_i in range(n_epochs):
            for batch_i in range(mnist.train.num_examples // batch_size):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={
                    x: batch_xs,
                    y: batch_ys
                })
            val_acc = sess.run(acc,
                        feed_dict={
                            x: mnist.validation.images,
                            y: mnist.validation.labels
                        })
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('%s - validation accuracy (epoch %d): %f' % (now, epoch_i, val_acc))
            if val_acc * 100 >= val_accuracy_target and not reached_target_accuracy:
                reached_target_accuracy = True
                print("reached validation accuracy target %.3f: %.4f (time %s)" % (val_accuracy_target, val_acc, str(time.time() - time_start)))

        # %% Print final test accuracy:
        evaluate_test_set_accuracy(acc)



def parse_args():
    parser = argparse.ArgumentParser(description='KungFu mnist example.')
    parser.add_argument(
        '--use-kungfu', type=bool, default=True, help='use kungfu optimizer')
    parser.add_argument(
        '--kungfu-strategy',
        type=str,
        default='plain', # Plain SyncSGD
        help='Specify KungFu strategy: \'plain\' or \'ako\' if --use-kungfu flag is set')
    parser.add_argument(
        '--ako-partitions', type=int, default=1, help='number of ako partitions')
    parser.add_argument(
        '--staleness', type=int, default=1, help='ako staleness constant')
    parser.add_argument(
        '--kickin-time', type=int, default=100, help='iteration starting from which ako kicks in')
    parser.add_argument(
        '--n-epochs', type=int, default=1, help='number of epochs')
    parser.add_argument(
        '--batch-size', type=int, default=50, help='batch size')
    parser.add_argument(
        '---val-accuracy-target', type=float, default=92., help='validation accuracy target')
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
    x, y_, train_step, acc = build_train_ops(args.use_kungfu, 
                                             args.kungfu_strategy, args.ako_partitions,
                                             args.staleness, args.kickin_time)
    show_info()
    measure(
        lambda: train_mnist(x, y_, train_step, acc, 
                            args.n_epochs, args.batch_size,
                            args.val_accuracy_target),
        'train')


measure(main, 'main')
