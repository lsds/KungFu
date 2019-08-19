import tensorflow as tf
from kungfu.ops import barrier, save_variable, request_variable


def test_barrier():
    with tf.Session() as sess:
        sess.run(barrier())


def test_save_and_request():
    global_step = tf.Variable(tf.constant(0, dtype=tf.int64))
    target = tf.Variable(tf.constant(0, dtype=tf.int32))

    x = tf.Variable(tf.zeros([10], dtype=tf.int32))

    inc_op = tf.assign_add(global_step, 1)
    update_op = tf.assign(x, x + 1)
    save_op = save_variable(global_step, x)
    y = request_variable(target, global_step, x.name, x.shape, x.dtype)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(barrier())

        for _ in range(3):
            sess.run([inc_op, update_op])
            sess.run(save_op)
            v = sess.run(y)
            print(v)

        sess.run(barrier())


# TODO: more tests

test_barrier()
test_save_and_request()
