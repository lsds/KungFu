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

cluster_schedule = [
    # (gs, new_size) :: scale cluster to new_size after step gs
    (3, 3),
    (5, 4),
]

propose_steps = [gs for gs, _ in cluster_schedule]
update_steps = [gs + 1 for gs, _ in cluster_schedule]


def compute_new_size(schedule, global_step, init):
    schedule = reversed(sorted(schedule))
    return tf.case([(tf.greater_equal(
        global_step, step), lambda size=size: tf.constant(size))
                    for step, size in schedule], lambda: tf.constant(init))


new_cluster_size = compute_new_size(cluster_schedule, global_step, 0)
propose_op = propose_update(global_step + 1, version, new_cluster_size)
update_op = update_cluster(version)

init = tf.global_variables_initializer()

max_step = 10

with tf.Session() as sess:
    sess.run(init)
    init_gs = sess.run(global_step)
    print('init step is %d' % (init_gs))

    while True:
        gs = sess.run(global_step)
        print('step: %d' % (gs))

        # BEGIN hook
        if gs in update_steps:
            sess.run(update_op)
        # END hook

        v = sess.run(y)
        # sess.run(fake_train_op)
        print(v)

        # BEGIN hook
        if gs in propose_steps:
            sess.run(advance_version)
            # all peers write to config server with idential content (the new cluster size)
            _, keep = sess.run(propose_op)
            if not keep:
                break
        # END hook

        gs = sess.run(advance_global_step)
        if gs >= max_step:
            break
