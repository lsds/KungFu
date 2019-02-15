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

#from official.mnist import dataset as mnist_dataset




from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

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

    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits

def build_train_ops(use_kungfu, kungfu_strategy, ako_partitions, staleness, kickin_time):
    # Parameters
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_step = 1

    x = tf.placeholder(tf.float32, shape=(None,32,32,1))
    y = tf.placeholder(tf.int32, (None)) # 0-9 digits recognition => 10 classes
    one_hot_y = tf.one_hot(y, 10)

    #Invoke LeNet function by passing features
    logits = LeNet5(x)

    #Softmax with cost function implementation
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)

    if use_kungfu:
        import kungfu as kf
        optimizer = kf.SyncSGDOptimizer(optimizer, strategy=kungfu_strategy,
                                      ako_partitions=ako_partitions,
                                      staleness=staleness,
                                      kickin_time=kickin_time)

    train_step = optimizer.minimize(loss, name='train_step')
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(one_hot_y,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y, train_step, acc


def train_mnist(x, y, train_step, acc, n_epochs, batch_size, val_accuracy_target):
    n_epochs = 25
    batch_size = 50

    def  evaluate_test_set_accuracy(acc):
        test_acc = sess.run(acc,
                    feed_dict={
                        x: X_test,
                        y: y_test
        })
        print('test accuracy: %f' % test_acc)

    time_start = time.time()
    reached_target_accuracy = False
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch_i in range(n_epochs):
            for offset in range(0, len(X_train), batch_size):
                batch_xs, batch_ys = X_train[offset:offset+batch_size], y_train[offset:offset+batch_size]
                sess.run(train_step, feed_dict={
                    x: batch_xs,
                    y: batch_ys
                })
            val_acc = sess.run(acc,
                        feed_dict={
                            x: X_validation,
                            y: y_validation
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
