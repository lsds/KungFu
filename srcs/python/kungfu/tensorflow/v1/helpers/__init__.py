import tensorflow as tf


def random_input(sample_shape, feature_categories):
    samples = tf.random_uniform(sample_shape)
    labels = tf.random_uniform([],
                               minval=0,
                               maxval=feature_categories - 1,
                               dtype=tf.int32)
    return samples, labels
