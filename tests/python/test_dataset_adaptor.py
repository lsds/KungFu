#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from kungfu.datasets.adaptor import ExampleDatasetAdaptor
from kungfu.helpers.cifar import Cifar10Loader, Cifar100Loader


def train_to_end(original_ds, adapt, handle):
    init_ds, get_next = adapt(original_ds)

    update_offset_op = adapt.create_update_offset()
    update_topology_op = adapt.create_update_topology()
    rewind_op = adapt.create_rewind()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        adapt.debug(sess)

        epoches = 20
        for epoch in range(epoches):
            print('BEGIN epoch')
            print('epoch %d' % epoch)
            sess.run(init_ds)
            adapt.debug(sess)

            step = 0
            while True:
                step += 1
                try:
                    v = sess.run(get_next)
                    print('epoch: %d, step: %d' % (epoch, step))
                    handle(v)
                    sess.run(update_offset_op)
                    adapt.debug(sess)
                except tf.errors.OutOfRangeError:
                    print('[W] rewind to beginning!')
                    sess.run(rewind_op)
                    sess.run(init_ds)
                    adapt.debug(sess)
                if step > 4:
                    break
            sess.run(update_topology_op)
            print('END epoch')
            print('\n')


def simple_example():
    n = 1024
    source = np.array(list(range(n)))
    ds = tf.data.Dataset.from_tensor_slices(source)
    adapt = ExampleDatasetAdaptor(batch_size=10, shard_count=4)

    def handle(v):
        print(v)

    train_to_end(ds, adapt, handle)


def cifar_example():
    # loader = Cifar10Loader()
    loader = Cifar100Loader()

    data = loader.load_datasets()
    images = tf.data.Dataset.from_tensor_slices(data.train.images)
    labels = tf.data.Dataset.from_tensor_slices(data.train.labels)
    ds = tf.data.Dataset.zip((images, labels))

    adapt = ExampleDatasetAdaptor(batch_size=5000, shard_count=4)

    def handle(v):
        xs, y_s = v
        print('xs :: %s, y_s :: %s' % (xs.shape, y_s.shape))

    train_to_end(ds, adapt, handle)


def main():
    simple_example()
    cifar_example()


main()
