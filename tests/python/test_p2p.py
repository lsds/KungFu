import time
import tensorflow as tf

import kungfu as kf
from kungfu import ops

import os
self_rank = int(os.getenv('KUNGFU_TEST_SELF_RANK'))

x = tf.ones((3, 3))
destination = tf.constant(1)
op = tf.cond(tf.not_equal(destination, self_rank), lambda: ops.send_to(destination, x), lambda: tf.no_op())


with tf.Session() as sess:
    sess.run(op)
    time.sleep(2)