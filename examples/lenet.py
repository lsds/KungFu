from __future__ import print_function

import argparse
import os
import time
import timeit
import sys

import random 

import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from kungfu.helpers.utils import show_size
from kungfu.helpers.mnist import load_datasets

import kungfu as kf

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



def LeNet5(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }

    
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 

    conv1 = tf.nn.relu(conv1)

    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b

    conv2 = tf.nn.relu(conv2)

    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    
    fc1 = flatten(pool_2)
    
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    
    fc1 = tf.nn.relu(fc1)

    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)
    
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits

def build_train_ops(kungfu_strategy, ako_partitions, device_batch_size):
    # Parameters
    learning_rate = 0.01
    training_epochs = 25
    display_step = 1

    x = tf.placeholder(tf.float32, shape=(None,32,32,1))
    y = tf.placeholder(tf.int32, (None, 10)) # 0-9 digits recognition => 10 classes
    #one_hot_y = tf.one_hot(y, 10)

    #Invoke LeNet function by passing features
    logits = LeNet5(x)

    #Softmax with cost function implementation
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = y) #one_hot_y)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

    if kungfu_strategy == 'ako':
        from kungfu.optimizers import AkoOptimizer
        optimizer = AkoOptimizer(optimizer, ako_partitions=ako_partitions)
    else:
        from kungfu.optimizers import ParallelOptimizer
        print("Using parallel optimizer")
        optimizer = ParallelOptimizer(optimizer, device_batch_size=device_batch_size)

    future_batch_op, train_step = optimizer.minimize(loss, name='train_step')
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y, train_step, acc, future_batch_op


def train_mnist(x, y, mnist, train_step, acc, n_epochs, local_batch, val_accuracy_target, future_batch_op, now):
    reached_target_accuracy = False
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("/home/ab7515/tensorboard-logs-andrei", sess.graph)

        sess.run(tf.global_variables_initializer())
        
        sess.run(kf.distributed_variables_initializer())

        time_start = time.time()
        total_val_duration = 0
        window = []
        img_secs = []
        for epoch_i in range(n_epochs):
            # Static batch
            def feed_batches_static(batch_size):
                mini_batch_indices = list(range(0, len(mnist.train.images), batch_size))
                random.shuffle(mini_batch_indices)

                for iteration_id, offset in enumerate(mini_batch_indices):
                    batch_xs, batch_ys = mnist.train.images[offset:offset+batch_size], mnist.train.labels[offset:offset+batch_size]
                    _, _ = sess.run([train_step, future_batch_op], feed_dict={
                        x: batch_xs,
                        y: batch_ys
                    })
                    val_acc = sess.run(acc,
                         feed_dict={
                             x: mnist.test.images,
                             y: mnist.test.labels
                    })
                    print('[%10.3f] - validation accuracy: %f' % (time.time() - now, val_acc))

            # Adaptive batches    
            def feed_batches(batch_size):
                mini_batch_indices = list(range(0, len(mnist.train.images), batch_size))
                random.shuffle(mini_batch_indices)
                interval = 10
                examples_processed = 0
                for iteration_id, offset in enumerate(mini_batch_indices):
                    batch_xs, batch_ys = mnist.train.images[offset:offset+batch_size], mnist.train.labels[offset:offset+batch_size]
                    _, future_batch_size = sess.run([train_step, future_batch_op], feed_dict={
                        x: batch_xs,
                        y: batch_ys
                    })
                    examples_processed += batch_size
                    val_acc = sess.run(acc,
                         feed_dict={
                             x: mnist.test.images,
                             y: mnist.test.labels
                    })
                    print('[%10.3f] - validation accuracy: %f' % (time.time() - now, val_acc))

                    if iteration_id % interval == 0:
                        if future_batch_size >= 10000:
                            continue
                        future_batch_size = int(future_batch_size)
                        print("Changing batch size from %d to %d" % (batch_size, future_batch_size))
                        return examples_processed, future_batch_size

            dynamic = False
            if dynamic:
                examples_processed = 0
                batch = local_batch
                while examples_processed <= 50000:
                    examples, future_batch_size = feed_batches(batch)
                    examples_processed += examples
                    batch = future_batch_size
            else:    
                timeEpoch = timeit.timeit(lambda: feed_batches_static(local_batch), number=1)
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
                print('%s - validation accuracy (epoch %d): %f' % (now, epoch_i, val_acc))

            # Results
            img_sec_mean = np.mean(img_secs)
            img_sec_conf = 1.96 * np.std(img_secs)
            print('Img/sec per CPU: %.2f +- %.2f' % (img_sec_mean, img_sec_conf))
            print('Total img/sec: %.2f +- %.2f' % (4 * img_sec_mean, 4 * img_sec_conf))


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
    parser.add_argument(
        '--use-kungfu', type=bool, default=True, help='use kungfu optimizer')
    parser.add_argument(
        '--kungfu-strategy',
        type=str,
        default='cpu_all_reduce', # KungFu Parallel SGD
        help='Specify KungFu strategy: \'plain\' or \'ako\' if --use-kungfu flag is set')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='',
        help='Specify mnist directory')
    parser.add_argument(
        '--ako-partitions', type=int, default=1, help='number of ako partitions')
    parser.add_argument(
        '--n-epochs', type=int, default=1, help='number of epochs')
    parser.add_argument(
        '--batch-size', type=int, default=50, help='batch size')
    parser.add_argument(
        '---val-accuracy-target', type=float, default=98.5, help='validation accuracy target')
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
    print('%d vars, total dim: %d, total size: %s' % (tot_vars, tot_dim,
                                                      show_size(tot_size)))


def warmup():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


def main():
    args = parse_args()
    measure(warmup, 'warmup')
    x, y_, train_step, acc, future_batch_op = build_train_ops(args.kungfu_strategy, args.ako_partitions, args.batch_size)
    show_trainable_variables_info()
    
    mnist = measure(lambda: load_datasets(args.data_dir, normalize=True, one_hot=True, padded=True), 'load data')

    measure(
        lambda: train_mnist(x, y_, mnist, train_step, acc, 
                            args.n_epochs, args.batch_size,
                            args.val_accuracy_target, future_batch_op, time.time()), 'train')


measure(main, 'main')
