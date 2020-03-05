import tensorflow as tf
from kungfu.tensorflow.ops import (broadcast, barrier, counter, group_all_reduce,
                                   peer_info, request_variable, save_variable)
from kungfu import current_rank

def test_broadcast():
    v = tf.Variable(True if current_rank() == 0 else False)
    u = broadcast(v)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x = sess.run(v)
        y = sess.run(u)
        print(x,y)

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

        for i in range(3):
            sess.run([inc_op, update_op])
            sess.run(save_op)
            sess.run(barrier())
            v = sess.run(y)
            assert v[0] == i + 1
        sess.run(barrier())


def test_consensus():
    from kungfu import current_cluster_size, current_rank
    from kungfu.tensorflow.ops import consensus

    np = current_cluster_size()
    rank = current_rank()

    x = tf.Variable(rank, dtype=tf.int32)
    consensus_check = consensus(x)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        v = sess.run(consensus_check)

        assert v == (np == 1)


# TODO: more tests


def test_all():
    test_barrier()
    test_group_all_reduce()
    test_peer_info()
    test_save_and_request()
    test_consensus()
    test_broadcast()


test_all()
