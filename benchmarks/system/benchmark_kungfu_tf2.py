#!/usr/bin/env python3
"""
Implemented based on:
https://github.com/uber/horovod/blob/master/examples/tensorflow2_synthetic_benchmark.py
"""
from __future__ import absolute_import, division, print_function

import argparse
import timeit

import numpy as np
import tensorflow as tf
from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.optimizers import (PairAveragingOptimizer,
                                          SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer)
from tensorflow.keras import applications

# Benchmark settings
parser = argparse.ArgumentParser(
    description='TensorFlow Synthetic Benchmark',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce',
                    action='store_true',
                    default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--model',
                    type=str,
                    default='ResNet50',
                    help='model to benchmark')
parser.add_argument('--batch-size',
                    type=int,
                    default=32,
                    help='input batch size')
parser.add_argument('--num-warmup-batches',
                    type=int,
                    default=10,
                    help='number of warm-up batches')
parser.add_argument('--num-batches-per-iter',
                    type=int,
                    default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters',
                    type=int,
                    default=10,
                    help='number of benchmark iterations')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--kf-optimizer',
                    type=str,
                    default='sync-sgd',
                    help='kungfu optimizer')

args = parser.parse_args()
args.cuda = not args.no_cuda

# Set up standard model.
model = getattr(applications, args.model)(weights=None)

# opt = tf.optimizers.SGD(0.01)
opt = tf.compat.v1.train.GradientDescentOptimizer(0.01)

# KungFu: wrap tf.compat.v1.train.Optimizer.
if args.kf_optimizer == 'sync-sgd':
    opt = SynchronousSGDOptimizer(opt)
elif args.kf_optimizer == 'async-sgd':
    opt = PairAveragingOptimizer(opt)
elif args.kf_optimizer == 'sma':
    opt = SynchronousAveragingOptimizer(opt)
else:
    raise RuntimeError('Unknown KungFu optimizer')

data = tf.random.uniform([args.batch_size, 224, 224, 3])
target = tf.random.uniform([args.batch_size, 1],
                           minval=0,
                           maxval=999,
                           dtype=tf.int64)


@tf.function
def benchmark_step(first_batch):

    with tf.GradientTape() as tape:
        probs = model(data, training=True)
        loss = tf.losses.categorical_crossentropy(target, probs)

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    if first_batch:
        from kungfu.tensorflow.initializer import broadcast_variables
        broadcast_variables(model.variables)
        broadcast_variables(opt.variables())


def log(s, nl=True):
    if current_rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, current_cluster_size()))

with tf.device(device):
    # Warm-up
    log('Running warmup...')
    benchmark_step(first_batch=True)
    timeit.timeit(lambda: benchmark_step(first_batch=False),
                  number=args.num_warmup_batches)

    # Benchmark
    log('Running benchmark...')
    img_secs = []
    for x in range(args.num_iters):
        time = timeit.timeit(lambda: benchmark_step(first_batch=False),
                             number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
        img_secs.append(img_sec)

    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
    log('Total img/sec on %d %s(s): %.1f +-%.1f' %
        (current_cluster_size(), device, current_cluster_size() * img_sec_mean,
         current_cluster_size() * img_sec_conf))
