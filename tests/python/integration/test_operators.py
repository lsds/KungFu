import tensorflow as tf
from kungfu.python import current_rank
from kungfu.tensorflow.ops import barrier
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


def test_broadcast():
    from kungfu.tensorflow.ops import broadcast
    v = tf.Variable(True if current_rank() == 0 else False)
    u = broadcast(v)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x = sess.run(v)
        y = sess.run(u)
        # print(x, y)
        assert (y == True)


def test_barrier():
    with tf.Session() as sess:
        sess.run(barrier())


def test_monitored_all_reduce():
    def gen_tree(n, r):
        tree = [i for i in range(n)]
        for i in range(n):
            if i != r:
                tree[i] = r
        return tree

    from kungfu.tensorflow.ops import monitored_all_reduce, current_cluster_size
    np = current_cluster_size()
    init_tree = gen_tree(np, 0)

    tree = tf.Variable(init_tree, dtype=tf.int32)
    x = tf.Variable(tf.ones([16, 1024, 1024], dtype=tf.int64))
    y = monitored_all_reduce(x, tree)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        v = sess.run(y)
        assert (v.sum() == np * 16 * 1024 * 1024)


def test_group_all_reduce():
    from kungfu.tensorflow.ops import group_all_reduce
    sizes = [i % 5 for i in range(10)]
    xs = [tf.Variable(tf.ones([n], tf.int32)) if n else None for n in sizes]
    ys = group_all_reduce(xs)
    op = [y for y in ys if y is not None]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(op)


def test_group_all_gather():
    from kungfu.python import current_cluster_size, current_rank
    from kungfu.tensorflow.ops import all_gather
    rank = current_rank()
    np = current_cluster_size()
    sizes = [i + 1 for i in range(5)]
    xs = [(rank + 1) * tf.Variable(tf.ones([n], tf.int32)) for n in sizes]
    ys = [all_gather(x) for x in xs]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i, y in enumerate(ys):
            v = sess.run(y)
            assert (v.sum() == (np + 1) * np / 2 * (i + 1))


def test_peer_info():
    from kungfu.tensorflow.ops import peer_info
    info = peer_info()
    with tf.Session() as sess:
        rank, np = sess.run(info)
        print('rank=%d, np=%d' % (rank, np))


def test_save_and_request():
    from kungfu.tensorflow.ops import request_variable, save_variable
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
    from kungfu.python import current_cluster_size, current_rank
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
    test_group_all_gather()
    test_group_all_reduce()
    test_peer_info()
    test_save_and_request()
    test_consensus()
    test_broadcast()
    test_monitored_all_reduce()


test_all()
