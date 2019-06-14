import tensorflow as tf
from kungfu.ops import barrier


def test_barrier():
    with tf.Session() as sess:
        sess.run(barrier())


# TODO: more tests

test_barrier()
