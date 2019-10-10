#!/usr/bin/env python

import argparse

import tensorflow as tf
from kungfu.ops import all_reduce, current_cluster_size
from kungfu.ops.adapt import get_init_checkpoint, resize_cluster
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

p = argparse.ArgumentParser(description='TF ML Benchmarks.')
p.add_argument('--schedule',
               type=str,
               default='3:2,3:4,3:16,3:1',
               help='cluster size schedule')

args = p.parse_args()


def parse_schedule(config):
    schedule = []
    t = 0
    for p in config.split(','):
        kv = p.split(':')
        n, val = (int(kv[0]), int(kv[1]))
        t += n
        schedule.append((t, val))
    return schedule, t


cluster_size_schedule, max_step = parse_schedule(args.schedule)
print(cluster_size_schedule)
print(max_step)


def get_new_size(i, sch, old):
    for j, n in reversed(sch):
        if i >= j:
            return n
    return old


x = tf.Variable(tf.ones([], dtype=tf.int32))
y = all_reduce(x)


def restore(checkpoint):
    gs = int(checkpoint)
    return gs


ckpt = tf.placeholder(tf.string)
new_size = tf.placeholder(tf.int32)
resize_op = resize_cluster(ckpt, new_size)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    init_gs = restore(get_init_checkpoint())
    print('restored to %d' % (init_gs))
    np = current_cluster_size()
    init_np = get_new_size(init_gs, cluster_size_schedule, np)
    if np != init_np:
        print(
            '[W] init cluster size (%d) is not consistent with schedule (%d)' %
            (np, init_np))

    for gs in range(init_gs, max_step):
        v = sess.run(y)
        print('step %d, result: %d' % (gs, v))

        next_gs = gs + 1
        np = get_new_size(next_gs, cluster_size_schedule, np)
        keep = sess.run(resize_op,
                        feed_dict={
                            ckpt: str(next_gs),
                            new_size: np
                        })
        if not keep:
            break
