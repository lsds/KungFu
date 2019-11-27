#!/usr/bin/env python3

import tensorflow as tf
from kungfu.tensorflow.ops import save_variables


def test_save_variables():
    w = tf.Variable(tf.ones([8, 3, 3, 32]))
    b = tf.Variable(tf.ones([32]))

    variables = [w, b]

    op = save_variables(variables)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            sess.run(op)


test_save_variables()
