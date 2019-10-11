#!/usr/bin/env python

import argparse
import time

t0 = time.time()  # before import tensorflow

import tensorflow as tf
from kungfu.ops import all_reduce, barrier, current_cluster_size
from kungfu.ops.adapt import get_init_checkpoint, resize_cluster
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


ckpt = tf.placeholder(tf.string)
new_size = tf.placeholder(tf.int32)
resize_op = resize_cluster(ckpt, new_size)

init = tf.global_variables_initializer()

# barrier_op = barrier()

with tf.Session() as sess:
    sess.run(init)

    init_gs = restore(get_init_checkpoint())
    np = current_cluster_size()
    init_np = get_cluster_size(init_gs, cluster_size_schedule, np)
    if np != init_np:
        print(
            '[W] init cluster size (np=%d) is not consistent with schedule (np=%d)'
            % (np, init_np))

    print('restored from %d, np=%d, init_np=%d, start took %s' %
          (init_gs, np, init_np, show_duration(time.time() - t0)))

    for gs in range(init_gs, max_step):
        t0 = time.time()
        v = sess.run(y)
        print('step %d, result: %d, np=%d, took %s' %
              (gs, v, np, show_duration(time.time() - t0)))

        next_gs = gs + 1
        if next_gs < max_step:
            new_np = get_cluster_size(next_gs, cluster_size_schedule, np)
            if new_np != np:
                t0 = time.time()
                keep = sess.run(resize_op,
                                feed_dict={
                                    ckpt: str(next_gs),
                                    new_size: new_np,
                                })
                print('resize %d -> %d took %s' %
                      (np, new_np, show_duration(time.time() - t0)))
                np = new_np
                if not keep:
                    break
