#!/usr/bin/env python3
# This example is inspired by https://www.tensorflow.org/guide/keras/train_and_evaluate
#
# KungFu requires users to make the following changes:
# 1. KungFu provides distributed optimizers that can wrap the original optimizer.
# The distributed optimizer defines how local gradients and model weights are synchronized.
# 2. (Optional) In a distributed training setting, the training dataset is often partitioned.
# 3. (Optional) Scaling the learning rate of your local optimizer
#
# Command to run this script:
# $ ./bin/kungfu-run -np 4 python3 examples/mnist_keras.py --n-epochs 10

import argparse
import logging

import kungfu as kf
import tensorflow as tf
from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.ops import broadcast
from kungfu.tensorflow.optimizers import (PairAveragingOptimizer,
                                          SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer)
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesCallback


def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # preprocess the mnist dataset
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    # create dataset
    dataset = dict()
    dataset['x_val'] = x_train[-10000:]
    dataset['y_val'] = y_train[-10000:]
    dataset['x_train'] = x_train[:-10000]
    dataset['y_train'] = y_train[:-10000]
    dataset['x_test'] = x_test
    dataset['y_test'] = y_test

    return dataset


def build_optimizer(name, n_shards=1):
    learning_rate = 0.1

    # Scale learning rate according to the level of data parallelism
    optimizer = tf.keras.optimizers.SGD(learning_rate=(learning_rate *
                                                       n_shards))

    # KUNGFU: Wrap the TensorFlow optimizer with KungFu distributed optimizers.
    if name == 'sync-sgd':
        return SynchronousSGDOptimizer(optimizer)
    elif name == 'async-sgd':
        return PairAveragingOptimizer(optimizer)
    elif name == 'sma':
        return SynchronousAveragingOptimizer(optimizer)
    else:
        raise RuntimeError('unknown optimizer: %s' % name)


def build_model(optimizer):
    num_classes = 10
    # create a model with keras
    model = tf.keras.Sequential()
    # add two hidden layer
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # add a dense layer with number of classes of nodes and softmax
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    # compile the model
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model


def train_model(model, dataset, n_epochs=1, batch_size=5000):
    n_shards = current_cluster_size()
    shard_id = current_rank()
    train_data_size = len(dataset['x_train'])

    # calculate the offset for the data of the KungFu node
    shard_size = train_data_size // n_shards
    offset = batch_size * shard_id

    # extract the data for learning of the KungFu node
    x = dataset['x_train'][offset:offset + shard_size]
    y = dataset['y_train'][offset:offset + shard_size]
    # train the model
    model.fit(x,
              y,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_data=(dataset['x_val'], dataset['y_val']),
              verbose=2,
              callbacks=[BroadcastGlobalVariablesCallback()])


def test_model(model, dataset):
    test_metrics = model.evaluate(dataset['x_test'],
                                  dataset['y_test'],
                                  verbose=0)
    # print test accuracy
    accuracy_index = 1
    print('test accuracy: %f' % test_metrics[accuracy_index])


def parse_args():
    parser = argparse.ArgumentParser(description='KungFu mnist example.')
    parser.add_argument('--kf-optimizer',
                        type=str,
                        default='sync-sgd',
                        help='available options: sync-sgd, async-sgd, sma')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=1,
                        help='number of epochs')
    parser.add_argument('--batch-size',
                        type=int,
                        default=50,
                        help='batch size')
    return parser.parse_args()


def main():
    logging.basicConfig(filename="tf2.log",
                        level=logging.DEBUG,
                        format="%(asctime)s:%(levelname)s:%(message)s")
    # parse arguments from the command line
    args = parse_args()
    # build the KungFu optimizer
    optimizer = build_optimizer(args.kf_optimizer)
    # build the Tensorflow model
    model = build_model(optimizer)
    # load mnist dataset
    dataset = load_dataset()
    # train the Tensorflow model
    train_model(model, dataset, args.n_epochs, args.batch_size)
    # test the performance of the Tensorflow model
    test_model(model, dataset)


if __name__ == '__main__':
    main()
