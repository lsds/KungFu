#!/usr/bin/env python3
import os
import tensorflow as tf

from kungfu.ops import update_cluster, barrier, all_reduce, propose_update

gs = tf.Variable(tf.zeros([], dtype=tf.int64))
inc_gs = tf.assign_add(gs, 1)
x = tf.Variable(tf.ones([], dtype=tf.int32))
op1 = all_reduce(x)
sync = barrier()
propose = propose_update(gs + 1)
update = update_cluster(gs)


def show_env():
    for k in os.environ:
        if k.startswith('KUNGFU_'):
            print('%s=%s' % (k, os.getenv(k)))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(sync)
    need_update = False
    for _ in range(10):
        show_env()
        sess.run(inc_gs)
        if need_update:
            exist = sess.run(update)
            if not exist:
                print('self NOT in the cluster anymore')
                break
        v = sess.run(op1)
        print(v)
        need_update = sess.run(propose)
