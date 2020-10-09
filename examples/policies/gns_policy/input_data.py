import kungfu.tensorflow as kf
import numpy as np
import tensorflow as tf

image_shape = {
    'ImageNet': [224, 224, 3],
    'CIFAR10': [32, 32, 3],
}


def batched_random_input(batch_size, sample_shape=None, feature_shape=None):
    if sample_shape is None:
        sample_shape = image_shape['CIFAR10']
    if feature_shape is None:
        feature_shape = []
    samples = tf.random_uniform([batch_size] + sample_shape)
    labels = tf.random_uniform([batch_size] + feature_shape,
                               minval=0,
                               maxval=999,
                               dtype=tf.int32)

    return samples, labels


def build_random_input_fn(init_batch_size, steps=None):
    def input_fn():
        batch_size = kf.get_or_create_batch_size(init_batch_size)
        samples, labels = batched_random_input(batch_size)
        samples = tf.data.Dataset.from_tensors(samples)
        labels = tf.data.Dataset.from_tensors(labels)
        features = {'x': samples}
        ds = tf.data.Dataset.zip((features, labels))
        ds = ds.repeat(steps)
        return ds

    return input_fn


def build_cifar10_input_fn(init_batch_size=32):
    def input_fn():
        samples, labels = tf.keras.datasets.cifar10.load_data()[0]
        samples = (samples / 255.0).astype(np.float32)
        labels = labels.astype(np.int32)
        samples = tf.data.Dataset.from_tensor_slices(samples)
        labels = tf.data.Dataset.from_tensor_slices(labels)
        features = {'x': samples}
        ds = tf.data.Dataset.zip((features, labels))
        ds = ds.repeat()
        batch_size = kf.get_or_create_batch_size(init_batch_size)
        ds = ds.batch(tf.cast(batch_size, tf.int64))
        return ds

    return input_fn
