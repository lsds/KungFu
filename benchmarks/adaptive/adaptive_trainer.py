#!/usr/bin/env python

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from kungfu.ops import all_reduce, current_cluster_size
from kungfu.ops.adapt import get_init_checkpoint, resize_cluster


def get_new_size(i, sch, old):
    for j, n in reversed(sch):
        if i >= j:
            return n
    return old


x = tf.Variable(tf.ones([], dtype=tf.int32))
y = all_reduce(x)

cluster_schedule = [
    # (gs, new_size) :: scale cluster to new_size after step gs
    (3, 3),
    (5, 4),
    (6, 2),
    #
    (8, 1),
    (10, 16),
    (13, 1),
]

max_step = 20  # 0, ..., 19,
# 14, 15, 16, 17, 18, 19

gs = 0


def restore(checkpoint):
    global gs
    gs = int(checkpoint)
    print('restored to %d' % (gs))


ckpt = tf.placeholder(tf.string)
new_size = tf.placeholder(tf.int32)
resize_op = resize_cluster(ckpt, new_size)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    restore(get_init_checkpoint())
    np = current_cluster_size()

    while True:
        v = sess.run(y)
        print('step %d, result: %d' % (gs, v))

        # BEGIN hook
        gs += 1
        np = get_new_size(gs, cluster_schedule, np)
        keep = sess.run(resize_op, feed_dict={ckpt: str(gs), new_size: np})
        if not keep:
            break
        if gs >= max_step:
            break
