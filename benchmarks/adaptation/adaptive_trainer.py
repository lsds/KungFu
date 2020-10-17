#!/usr/bin/env python

import argparse
import time

t0 = time.time()  # before import tensorflow

import tensorflow as tf
from kungfu.tensorflow.ops import (all_reduce, barrier, current_cluster_size,
                                   resize_cluster_from_url)
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

p = argparse.ArgumentParser(description='Adaptation Benchmark.')
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
        schedule.append((t, t + n, val))
        t += n
    return schedule, t


def show_duration(duration):
    if duration < 1:
        return '%.2fms' % (duration * 1e3)
    if duration < 60:
        return '%.2fs' % duration
    sec = int(duration)
    mm, ss = sec / 60, sec % 60
    if duration < 3600:
        return '%dm%ds' % (mm, ss)
    return '%dh%dm%ds' % (mm / 60, mm % 60, ss)


cluster_size_schedule, max_step = parse_schedule(args.schedule)
# print(cluster_size_schedule)
# print(max_step)


def get_cluster_size(i, sch, old):
    for s, e, n in sch:
        if s <= i and i < e:
            return n
    print('[W] not scheduled for %d' % (i))
    return old


x = tf.Variable(tf.ones([], dtype=tf.int32))
y = all_reduce(x)


def restore(checkpoint):
    gs = int(checkpoint)
    return gs


new_size = tf.placeholder(tf.int32)
resize_op = resize_cluster_from_url()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    np = current_cluster_size()

    gs_place = tf.placeholder(dtype=tf.int64, shape=())
    sync_step_op = all_reduce(gs_place, op='max')
    shoud_sync = True
    gs = 0
    while gs < max_step:
        if shoud_sync:
            new_gs = sess.run(sync_step_op, feed_dict={gs_place: gs})
            print('sync step: %d -> %d' % (gs, new_gs))
            gs = new_gs
            shoud_sync = False
        t0 = time.time()
        v = sess.run(y)
        print('step %d, result: %d, np=%d, took %s' %
              (gs, v, np, show_duration(time.time() - t0)))

        next_gs = gs + 1
        if next_gs < max_step:
            new_np = get_cluster_size(next_gs, cluster_size_schedule, np)
            if new_np != np:
                t0 = time.time()
                print('TODO: propose new np: %d' % (np))
                changed, detached = sess.run(resize_op)
                print('resize %d -> %d took %s' %
                      (np, new_np, show_duration(time.time() - t0)))
                np = new_np
                if detached:
                    break
                if changed:
                    shoud_sync = True
        gs += 1
