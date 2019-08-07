#!/usr/bin/env python3
import os
import tensorflow as tf

from kungfu.ops import update_cluster, barrier, all_reduce, propose_update

init_sess = os.getenv('KUNGFU_INIT_SESS')
init_gs = int(init_sess) if init_sess else 0

gs = tf.Variable(tf.constant(init_gs, dtype=tf.int64))
inc_gs = tf.assign_add(gs, 1)
x = tf.Variable(tf.ones([], dtype=tf.int32))
op1 = all_reduce(x)
sync = barrier()
best_size = tf.Variable(tf.ones([], dtype=tf.int32))
propose = propose_update(gs + 1, best_size)
update = update_cluster(gs)

host_cap = 4


def compute_new_size(prev_size):
    # return tf.mod(prev_size, host_cap) + 1
    return tf.cast(gs + 1, tf.int32)


compute_new_size = tf.assign(best_size, compute_new_size(best_size))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    need_update = False
    while True:
        step = sess.run(gs)
        if step > host_cap:
            break

        if need_update:
            exist = sess.run(update)
            if not exist:
                print('self NOT in the cluster anymore')
                break
        v = sess.run(op1)
        print(v)
        sess.run(compute_new_size)

        if step < host_cap:
            need_update = sess.run(propose)

        sess.run(inc_gs)
