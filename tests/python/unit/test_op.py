import tensorflow as tf

from kungfu.tensorflow.ops import counter, step_based_schedule


def test_counter():
    c = counter()
    with tf.Session() as sess:
        for i in range(10):
            n = sess.run(c)
            assert (n == i)


def test_step_based_scheduler():
    sizes = [1, 2, 4, 8]
    n_step = 3
    config = ','.join('%d:%d' % (size, n_step) for size in sizes)
    expected_sizes = [1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8]
    schedule = step_based_schedule(config)
    with tf.Session() as sess:
        for i in range(12):
            size = sess.run(schedule)
            assert (size == expected_sizes[i])
