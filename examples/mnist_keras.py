#!/usr/bin/env python3
# This example shows how a MNIST Single Layer Perception Model training program 
# can adopt various distributed synchronization strategies using KungFu.
# 
# In principle, KungFu requires users to make three changes:
# 1. KungFu provides distributed optimizers that can wrap the original optimizer. 
# The distributed optimizer defines how local gradients and model weights are synchronized. 
# 2. KungFu provides distributed variable initializers that defines how model weights are
# initialized on distributed devices.
# 3. (Optional) In a distributed training setting, the training dataset is often partitioned.  

import argparse
import os

import tensorflow as tf

import kungfu as kf
from kungfu.ops import current_cluster_size, current_rank


def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # preprocess the mnist dataset
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    # create dataset
    dataset = dict()
    dataset['x_train'] = x_train
    dataset['x_test'] = x_test
    dataset['y_train'] = y_train
    dataset['y_test'] = y_test

    return dataset


def build_optimizer(name, n_shards=1):
    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate / n_shards)

    # KUNGFU: Wrap the TensorFlow optimizer with KungFu distributed optimizers.
    if name == 'sync-sgd':
        from kungfu.optimizers import SyncSGDOptimizer
        return SyncSGDOptimizer(optimizer)
    if name == 'variance':
        from kungfu.optimizers import SyncSGDWithGradVarianceOptimizer
        return SyncSGDWithGradVarianceOptimizer(optimizer, monitor_interval=10)
    elif name == 'model-avg':
        from kungfu.optimizers import PeerModelAveragingOptimizer
        return PeerModelAveragingOptimizer(optimizer)
    else:
        raise RuntimeError('unknow optimizer: %s' % name)


def build_model(optimizer):
    num_classes = 10
    # create a model with keras
    model = tf.keras.Sequential()
    # add a dense layer with number of classes of nodes and softmax
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    # compile the model
    model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return model


def train_model(model, dataset, n_epochs=1, batch_size=5000):
    n_shards = current_cluster_size()
    shard_id = current_rank()
    train_data_size = len(dataset['x_train'])

    shard_size = train_data_size // n_shards
    step_per_epoch = shard_size // batch_size
    n_steps = step_per_epoch * n_epochs
    print('step_per_epoch: %d, %d steps in total' % (step_per_epoch, n_steps))

    # calculate the offset for the data of the KungFu node
    offset = batch_size * shard_id
    # extract the data for learning of the KungFu node
    x = dataset['x_train'][offset:offset + batch_size, :]
    y = dataset['y_train'][offset:offset + batch_size]
    # train the model
    model.fit(x, y, batch_size=batch_size, epochs=n_epochs)


def test_model(model, dataset):
    test_metrics = model.evaluate(dataset['x_test'], dataset['y_test'])
    # print test accuracy
    accuracy_index = 1
    print('test accuracy: %f' % test_metrics[accuracy_index])


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
    # parse arguements from the command line
    args = parse_args()
    # build the KungFu optimizer
    optimizer = build_optimizer(args.optimizer)
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
