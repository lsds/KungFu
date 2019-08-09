import tensorflow as tf
from kungfu.ops import has_member

x = tf.Variable(tf.constant([1, 2, 3, 4], dtype=tf.int32))
e1 = tf.Variable(tf.constant(1, dtype=tf.int32))
e2 = tf.Variable(tf.constant(5, dtype=tf.int32))


def test_has_member():
    b1 = has_member(x, e1)
    b2 = has_member(x, e2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        v1 = sess.run(b1)
        v2 = sess.run(b2)
        print(v1, v2)


# TODO: more tests

test_has_member()
