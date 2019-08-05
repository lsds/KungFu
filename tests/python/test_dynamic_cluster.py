#!/usr/bin/env python3
import os
import tensorflow as tf

from kungfu.ops import update_cluster, barrier, all_reduce, propose_update

gs = tf.Variable(tf.zeros([], dtype=tf.int64))
inc_gs = tf.assign_add(gs, 1)
x = tf.Variable(tf.ones([], dtype=tf.int32))
op1 = all_reduce(x)
sync = barrier()
best_size = tf.Variable(tf.ones([], dtype=tf.int32))
propose = propose_update(gs + 1, best_size)
update = update_cluster(gs)

host_cap = 4


def compute_new_size(prev_size):
    return tf.mod(prev_size, host_cap) + 1


compute_new_size = tf.assign(best_size, compute_new_size(best_size))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(sync)
    need_update = False
    for _ in range(10):
        sess.run(inc_gs)
        if need_update:
            exist = sess.run(update)
            if not exist:
                print('self NOT in the cluster anymore')
                break
        v = sess.run(op1)
        print(v)
        sess.run(compute_new_size)

        need_update = sess.run(propose)
