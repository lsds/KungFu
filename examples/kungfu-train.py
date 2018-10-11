#!/usr/bin/env python3

import os
import time
import sys

from tensorflow.examples.tutorials.mnist import input_data  # FIXME: deprecated
import tensorflow as tf

# FIXME: install kungfu and load it from standard location
sys.path.append('.')

from kungfu.negotiator import AsyncSGDOptimizer

tf.logging.set_verbosity(tf.logging.ERROR)  # disable deprecation warning


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


def slp(imput_size, class_number):
    x = tf.placeholder(tf.float32, [None, imput_size])
    W = tf.Variable(tf.zeros([imput_size, class_number]))
    b = tf.Variable(tf.zeros([class_number]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return x, y


def build_train_ops():
    learning_rate = 0.5
    x, y = slp(28 * 28, 10)
    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    optmizer = tf.train.GradientDescentOptimizer(learning_rate)
    # use_async_sgd = False
    use_async_sgd = True
    if use_async_sgd:
        optmizer = AsyncSGDOptimizer(optmizer)
    train_step = optmizer.minimize(loss, name='train_step')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y_, train_step, acc


def train(x, y_, train_step, acc, dataset):
    train_data_size = 55000
    batch_size = 5000
    n_epochs = 10
    step_per_epoch = train_data_size // batch_size
    n_steps = step_per_epoch * n_epochs
    print('step_per_epoch: %d, %d steps in total' % (step_per_epoch, n_steps))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(1, n_steps + 1):
            print('step: %d' % step)
            xs, y_s = dataset.train.next_batch(batch_size)
            sess.run(train_step, {x: xs, y_: y_s})
            if step % step_per_epoch == 0:
                result = sess.run(acc, {
                    x: dataset.validation.images,
                    y_: dataset.validation.labels
                })
                print('validation accuracy: %f' % result)

        result = sess.run(acc, {
            x: dataset.test.images,
            y_: dataset.test.labels
        })
        print('test accuracy: %f' % result)


def main():
    x, y_, train_step, acc = build_train_ops()
    data_dir = os.path.join(os.getenv('HOME'), 'var/data/mnist')
    mnist = measure(lambda: input_data.read_data_sets(data_dir, one_hot=True),
                    'load data')
    measure(lambda: train(x, y_, train_step, acc, mnist), 'train')


measure(main, 'main')
