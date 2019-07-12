import argparse
import os
import time
import timeit
import sys

import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from kungfu.helpers.utils import show_size
from kungfu.helpers.mnist import load_datasets

import kungfu as kf


def get_number_of_trainable_parameters():
    return np.sum(
        [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


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


# Source: https://www.kaggle.com/danyfang/mnist-competition
def LeNet5(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {'layer_1': 6, 'layer_2': 16, 'layer_3': 120, 'layer_f1': 84}

    conv1_w = tf.Variable(
        tf.truncated_normal(shape=[5, 5, 1, 6], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1],
                         padding='VALID') + conv1_b

    conv1 = tf.nn.relu(conv1)

    pool_1 = tf.nn.max_pool(conv1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='VALID')

    conv2_w = tf.Variable(
        tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(
        pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    conv2 = tf.nn.relu(conv2)

    pool_2 = tf.nn.max_pool(conv2,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='VALID')

    fc1 = flatten(pool_2)

    fc1_w = tf.Variable(
        tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b

    fc1 = tf.nn.relu(fc1)

    fc2_w = tf.Variable(
        tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)

    fc3_w = tf.Variable(
        tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits


def build_train_ops(kungfu_strategy, ako_partitions, device_batch_size):
    # Parameters
    learning_rate = 0.01
    training_epochs = 25
    display_step = 1

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
    y = tf.placeholder(tf.int32,
                       (None, 10))  # 0-9 digits recognition => 10 classes
    #one_hot_y = tf.one_hot(y, 10)

    #Invoke LeNet function by passing features
    logits = LeNet5(x)

    #Softmax with cost function implementation
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=y)  #one_hot_y)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)

    if kungfu_strategy == 'p2p':
        from kungfu.optimizers import ModelAveragingOptimizer
        print("Using ModelAveragingOptimizer")
        optimizer = ModelAveragingOptimizer(optimizer)
    else:
        from kungfu.optimizers import SyncSGDOptimizer
        print("Using SyncSGDOptimizer")
        optimizer = SyncSGDOptimizer(optimizer)

    train_step = optimizer.minimize(loss, name='train_step')
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y, train_step, acc


def train_mnist(x, y, mnist, train_step, acc, n_epochs, n_batches, batch_size,
                val_accuracy_target, kungfu_strategy):
    reached_target_accuracy = False
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("/home/ab7515/tensorboard-logs-andrei",
                                       sess.graph)

        sess.run(tf.global_variables_initializer())

        initializer = None
        if kungfu_strategy == 'p2p':
            from kungfu.optimizers import ModelAveragingOptimizer
            initializer = ModelAveragingOptimizer.get_initializer()
        else:
            initializer = kf.distributed_variables_initializer()

        sess.run(initializer)

        time_start = time.time()
        total_val_duration = 0
        window = []
        img_secs = []
        for epoch_i in range(n_epochs):

            def feed_batches(n_batches):
                if n_batches == -1:
                    n_batches = len(mnist.train.images)
                for offset in range(0, n_batches, batch_size):
                    batch_xs, batch_ys = mnist.train.images[
                        offset:offset +
                        batch_size], mnist.train.labels[offset:offset +
                                                        batch_size]
                    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

            # Measure throughput
            timeEpoch = timeit.timeit(lambda: feed_batches(n_batches),
                                      number=1)
            img_sec = len(mnist.train.images) / timeEpoch
            print('Epoch #%d: %.1f img/sec per CPU' % (epoch_i, img_sec))
            img_secs.append(img_sec)

            before_validation = time.time()
            val_acc = sess.run(acc,
                               feed_dict={
                                   x: mnist.test.images,
                                   y: mnist.test.labels
                               })
            after_validation = time.time()
            total_val_duration += after_validation - before_validation

            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('%s - validation accuracy (epoch %d): %f' %
                  (now, epoch_i, val_acc))

            window_val_acc_median = 0
            if not reached_target_accuracy:
                window.append(val_acc)
                if len(window) > 1:
                    window.pop(0)

                window_val_acc_median = 0 if len(window) < 1 else np.median(
                    window)
                if window_val_acc_median * 100 >= val_accuracy_target:
                    reached_target_accuracy = True
                    print(
                        "reached validation accuracy target %.3f: %.4f (time %s)"
                        % (val_accuracy_target, val_acc,
                           str(time.time() - time_start - total_val_duration)))

        # Results
        img_sec_mean = np.mean(img_secs)
        img_sec_conf = 1.96 * np.std(img_secs)
        print('Img/sec per CPU: %.2f +- %.2f' % (img_sec_mean, img_sec_conf))
        print('Total img/sec: %.2f +- %.2f' %
              (4 * img_sec_mean, 4 * img_sec_conf))

        # %% Print final test accuracy:
        test_acc = sess.run(acc,
                            feed_dict={
                                x: mnist.test.images,
                                y: mnist.test.labels
                            })
        print('test accuracy: %f' % test_acc)
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='KungFu mnist example.')
    parser.add_argument('--use-kungfu',
                        type=bool,
                        default=True,
                        help='use kungfu optimizer')
    parser.add_argument(
        '--kungfu-strategy',
        type=str,
        default='sync_sgd',
        help=
        'Specify KungFu strategy: \'sync_sgd\' or \'p2p\' if --use-kungfu flag is set'
    )
    parser.add_argument('--ako-partitions',
                        type=int,
                        default=1,
                        help='number of ako partitions')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=1,
                        help='number of epochs')
    parser.add_argument('--n-batches',
                        type=int,
                        default=-1,
                        help='number of batches')
    parser.add_argument('--batch-size',
                        type=int,
                        default=50,
                        help='batch size')
    parser.add_argument('---val-accuracy-target',
                        type=float,
                        default=98.5,
                        help='validation accuracy target')
    return parser.parse_args()


def show_trainable_variables_info():
    g = tf.get_default_graph()
    tot_vars = 0
    tot_dim = 0
    tot_size = 0
    for v in g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        dim = v.shape.num_elements()
        tot_vars += 1
        tot_dim += dim
        tot_size += dim * v.dtype.size
    print('%d vars, total dim: %d, total size: %s' %
          (tot_vars, tot_dim, show_size(tot_size)))


def main():
    args = parse_args()
    x, y_, train_step, acc = build_train_ops(args.kungfu_strategy,
                                             args.ako_partitions,
                                             args.batch_size)
    show_trainable_variables_info()

    mnist = measure(
        lambda: load_datasets(
            './mnist', normalize=True, one_hot=True, padded=True), 'load data')

    measure(
        lambda: train_mnist(x, y_, mnist, train_step, acc, args.n_epochs, args.
                            n_batches, args.batch_size, args.
                            val_accuracy_target, args.kungfu_strategy),
        'train')


measure(main, 'main')
