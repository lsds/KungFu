import tensorflow as tf
from kungfu.ops import adaptive_request_variables, barrier, save_variables


def test_adaptive_request_variables():
    w = tf.Variable(tf.ones([8, 3, 3, 32]))
    b = tf.Variable(tf.ones([32]))
    variables = [w, b]

    with tf.control_dependencies([save_variables(variables)]):
        init = barrier()

    op = adaptive_request_variables(variables)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init)
        for _ in range(100):
            sess.run(op)


test_adaptive_request_variables()
