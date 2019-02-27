#!/usr/bin/env python3

print('before import tf')
import tensorflow as tf

print('before import kungfu')
from kungfu.ops import _lazy_load_op_lib

print('before lazy_load_op_lib')
_op_lib = _lazy_load_op_lib()


def test_op_names():
    print('test_op_names')
    for d in dir(_op_lib):
        print(d)


def test_tf_session():
    print('test_tf_session')
    g = tf.Variable(tf.ones(shape=(2, 2)))
    # a = _op_lib.all_reduce_gpu(g)
    a = _op_lib.all_reduce(g)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(a)


def test_all():
    test_op_names()


test_all()
