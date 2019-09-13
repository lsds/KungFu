#!/usr/bin/env python

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from kungfu.ops import propose_update, update_cluster, all_reduce, get_init_version, get_start_step


def _create_version_tensor():
    init_version = get_init_version()
    version_tensor = tf.Variable(tf.constant(init_version, tf.int32),
                                 trainable=False)
    return version_tensor


x = tf.Variable(tf.ones([], dtype=tf.int32))
y = all_reduce(x)

version = _create_version_tensor()
advance_version = tf.assign_add(version, 1)

global_step = tf.Variable(get_start_step(version), trainable=False)
advance_global_step = tf.assign_add(global_step, 1)

new_cluster_size = tf.Variable(3 * tf.ones([], dtype=tf.int32))
propose_op = propose_update(global_step + 1, version, new_cluster_size)
update_op = update_cluster(version)

init = tf.global_variables_initializer()

max_step = 10

# change size 2 -> 3 at gs = 3
# gs : 1 2 3 4 5 6 7 8 9 10
#          *
# np : 2 2 2 3 3 3 3 3 3 3

with tf.Session() as sess:
    sess.run(init)
    init_gs = sess.run(global_step)
    print('init step is %d' % (init_gs))

    while True:
        gs = sess.run(global_step)
        print('step: %d' % (gs))

        if gs == 4:
            sess.run(update_op)

        v = sess.run(y)
        # sess.run(fake_train_op)
        print(v)

        if gs == 3:
            sess.run(advance_version)
            # all peers write to config server with idential content (the new cluster size)
            _, keep = sess.run(propose_op)
            if not keep:
                break

        gs = sess.run(advance_global_step)
        if gs >= max_step:
            break
