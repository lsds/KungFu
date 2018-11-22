#!/usr/bin/env python3

# This small script is used for checking if cuda/cudnn are installed properly

import tensorflow as tf

x = tf.Variable(tf.ones([3, 3, 3]))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y = sess.run(x)
    print(y)

# TODO: add more checks
