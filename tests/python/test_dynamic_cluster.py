#!/usr/bin/env python3
import argparse

import os
import tensorflow as tf

from kungfu.ops import update_cluster, barrier, all_reduce, propose_update


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--max-peer', type=int, default=8, help='')
    parser.add_argument('--max-step', type=int, default=24, help='')
    return parser.parse_args()


args = parse_args()

init_sess = os.getenv('KUNGFU_INIT_SESS')
init_gs = int(init_sess)

gs = tf.Variable(tf.constant(init_gs, dtype=tf.int64))
inc_gs = tf.assign_add(gs, 1)
x = tf.Variable(tf.ones([], dtype=tf.int32))
op1 = all_reduce(x)
# sync = barrier()
best_size = tf.Variable(tf.ones([], dtype=tf.int32))
propose = propose_update(gs + 1, best_size)
update = update_cluster(gs)

version = tf.Variable(tf.constant(init_sess, dtype=tf.string))


def compute_new_size(prev_size):
    return tf.mod(prev_size, args.max_peer) + 1
    # return tf.cast(gs + 1, tf.int32)


# kf_init = Opt.get_init()

compute_new_size = tf.assign(best_size, compute_new_size(best_size))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    need_update = True
    # FIXME: run broadcast ->
    # sess.run(kf_init) ->
    while True:
        step = sess.run(gs)
        if step > args.max_step:
            break

        if need_update:
            exist = sess.run(update)
            # FIXME: run broadcast <-
            # sess.run(kf_init) <-
            # - sync global step
            # - sync vars

            if not exist:
                print('self NOT in the cluster anymore')
                break

        # begin USER section
        v = sess.run(op1)  # TASK
        print(v)
        # end USER section

        sess.run(compute_new_size)

        if step < args.max_step:
            need_update = sess.run(propose)

        sess.run(inc_gs)
