import tensorflow as tf


def test_counter():
    from kungfu.tensorflow.ops import counter
    c = counter()
    with tf.Session() as sess:
        for i in range(10):
            n = sess.run(c)
            assert (n == i)


def test_counter_init():
    from kungfu.tensorflow.ops import counter
    c = counter(init=1)
    with tf.Session() as sess:
        for i in range(10):
            n = sess.run(c)
            assert (n == i + 1)


def test_exponential_moving_average():
    from kungfu.tensorflow.ops import exponential_moving_average as ema
    x = tf.Variable(1.0)
    y = ema(x, alpha=0.9)
    inc_x = tf.assign_add(x, 1)

    expected_vy = [2.0, 2.1, 2.29, 2.561, 2.9049]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(5):
            sess.run(inc_x)
            vy = sess.run(y)
            assert (abs(vy - expected_vy[i]) < 1e-6)


def test_step_based_scheduler():
    from kungfu.tensorflow.ops import step_based_schedule
    sizes = [1, 2, 4, 8]
    n_step = 3
    config = ','.join('%d:%d' % (size, n_step) for size in sizes)
    expected_sizes = [1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8]
    schedule = step_based_schedule(config)
    with tf.Session() as sess:
        for i in range(12):
            size = sess.run(schedule)
            assert (size == expected_sizes[i])


def test_detached():
    from kungfu.python import detached
    assert (not detached())


def test_rank():
    from kungfu.tensorflow.ops import rank
    rank_op = rank()
    with tf.Session() as sess:
        v = sess.run(rank_op)
        assert (v == 0)


def test_cluster_size():
    from kungfu.tensorflow.ops import cluster_size
    size_op = cluster_size()
    with tf.Session() as sess:
        v = sess.run(size_op)
        assert (v == 1)
