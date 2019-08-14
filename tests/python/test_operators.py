import tensorflow as tf
from kungfu.ops import barrier, peer_info


def test_barrier():
    with tf.Session() as sess:
        sess.run(barrier())


def test_peer_info():
    info = peer_info(tf.constant(-1, dtype=tf.int32))
    with tf.Session() as sess:
        rank, np = sess.run(info)
        print('rank=%d, np=%d' % (rank, np))


# TODO: more tests

test_barrier()
test_peer_info()
