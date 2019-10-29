import tensorflow as tf
from kungfu.tensorflow.v1.ops import barrier, group_all_reduce, peer_info, request_variable, save_variable


def test_barrier():
    with tf.Session() as sess:
        sess.run(barrier())


def test_group_all_reduce():
    sizes = [i % 5 for i in range(10)]
    xs = [tf.Variable(tf.ones([n], tf.int32)) if n else None for n in sizes]
    ys = group_all_reduce(xs)
    op = [y for y in ys if y is not None]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(op)


def test_peer_info():
    info = peer_info()
    with tf.Session() as sess:
        rank, np = sess.run(info)
        print('rank=%d, np=%d' % (rank, np))


def test_save_and_request():
    global_step = tf.Variable(tf.constant(0, dtype=tf.int64))
    target = tf.Variable(tf.constant(0, dtype=tf.int32))

    x = tf.Variable(tf.zeros([10], dtype=tf.int32))

    inc_op = tf.assign_add(global_step, 1)
    update_op = tf.assign(x, x + 1)
    save_op = save_variable(x, version=global_step)
    y = request_variable(target, global_step, x.name, x.shape, x.dtype)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(3):
            sess.run([inc_op, update_op])
            sess.run(save_op)
            sess.run(barrier())
            v = sess.run(y)
            print(v)
        sess.run(barrier())


# TODO: more tests

test_barrier()
test_group_all_reduce()
test_peer_info()
test_save_and_request()
