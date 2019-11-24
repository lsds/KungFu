import tensorflow as tf

from kungfu.tensorflow.ops import counter, step_based_schedule


def test_counter():
    c = counter()
    with tf.Session() as sess:
        for i in range(10):
            n = sess.run(c)
            assert (n == i)


def test_step_based_scheduler():
    config = '1'
    schedule = step_based_schedule(config)
    with tf.Session() as sess:
        for i in range(10):
            size = sess.run(schedule)
            assert (size == 1)
