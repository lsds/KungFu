#!/usr/bin/env python3

import json
import os
import time

import tensorflow as tf
from kungfu.tensorflow.ops import group_all_reduce

from resnet50 import grad_sizes


def fake_get_shard_info():
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    rank = int(os.getenv('KUNGFU_TEST_SELF_RANK'))
    cluster_size = len(cluster_spec['Peers'])
    return rank, cluster_size


def gen_fake_train_op(sizes):
    grads = []
    for size in sizes:
        grads.append(tf.Variable(tf.ones(shape=(size, ), dtype=tf.float32)))
    new_grads = group_all_reduce(grads)
    ops = []
    for g, new_g in zip(grads, new_grads):
        ops.append(tf.assign(g, new_g))
    return tf.group(ops)


def logEstimatedSpeed(batches, batchSize, dur, np):
    imgPerSec = batches * batchSize / dur
    print('Img/sec %.2f per worker, Img/sec %.2f per cluster, np=%d' %
          (imgPerSec, imgPerSec * np, np))


def fake_train(fake_train_op):
    rank, np = fake_get_shard_info()
    n_iters = 11
    steps_per_iter = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t0 = time.time()
        step = 0
        for _ in range(n_iters):
            for _ in range(steps_per_iter):
                step += 1
                sess.run(fake_train_op)
            if rank == 0:
                print('after %d steps' % step)
        d = time.time() - t0
        if rank == 0:
            logEstimatedSpeed(n_iters * steps_per_iter, 32, d, np)


fake_train_op = gen_fake_train_op(grad_sizes)
fake_train(fake_train_op)
