#!/usr/bin/env python3
"""
Implemented based on:
https://github.com/uber/horovod/blob/master/examples/tensorflow_synthetic_benchmark.py
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import timeit

import numpy as np
import tensorflow as tf
from tensorflow.keras import applications

# Benchmark settings
parser = argparse.ArgumentParser(
    description='TensorFlow Synthetic Benchmark',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model',
                    type=str,
                    default='ResNet50',
                    help='model to benchmark')
parser.add_argument('--batch-size',
                    type=int,
                    default=32,
                    help='input batch size')
parser.add_argument(
    '--num-warmup-batches',
    type=int,
    default=10,
    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter',
                    type=int,
                    default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters',
                    type=int,
                    default=10,
                    help='number of benchmark iterations')
parser.add_argument('--eager',
                    action='store_true',
                    default=False,
                    help='enables eager execution')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--kf-optimizer',
                    type=str,
                    default='sync-sgd',
                    help='KungFu optimizers')
parser.add_argument('--optimizer',
                    type=str,
                    default='sgd',
                    help='Optimizer: sgd, adam')
parser.add_argument('--fuse',
                    action='store_true',
                    default=False,
                    help='Fuse KungFu operations')

args = parser.parse_args()
args.cuda = not args.no_cuda

config = tf.ConfigProto()
if args.cuda:
    config.gpu_options.allow_growth = True
    from kungfu.ext import _get_cuda_index
    config.gpu_options.visible_device_list = str(_get_cuda_index())
else:
    config.gpu_options.allow_growth = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config.gpu_options.visible_device_list = ''

if args.eager:
    tf.enable_eager_execution(config)

# Set up standard model.
model = getattr(applications, args.model)(weights=None)

opt = None
learning_rate = 0.01
if args.optimizer == 'sgd':
    opt = tf.train.GradientDescentOptimizer(learning_rate)
elif args.optimizer == 'adam':
    opt = tf.train.AdamOptimizer(learning_rate)
else:
    raise Exception('Unknown optimizer option')

barrier_op = None

if args.kf_optimizer:
    from kungfu.tensorflow.ops import barrier
    barrier_op = barrier()
    if args.kf_optimizer == 'sync-sgd':
        from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
        opt = SynchronousSGDOptimizer(opt)
    elif args.kf_optimizer == 'sync-sgd-nccl':
        from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
        opt = SynchronousSGDOptimizer(opt, nccl=True, nccl_fusion=args.fuse)
    elif args.kf_optimizer == 'async-sgd':
        from kungfu.tensorflow.optimizers import PairAveragingOptimizer
        opt = PairAveragingOptimizer(opt, fuse_requests=args.fuse)
    elif args.kf_optimizer == 'sma':
        from kungfu.tensorflow.optimizers import SynchronousAveragingOptimizer
        opt = SynchronousAveragingOptimizer(opt)
    else:
        raise Exception('Unknown kungfu option')

data = tf.random_uniform([args.batch_size, 224, 224, 3])
target = tf.random_uniform([args.batch_size, 1],
                           minval=0,
                           maxval=999,
                           dtype=tf.int64)


def loss_function():
    logits = model(data, training=True)
    return tf.losses.sparse_softmax_cross_entropy(target, logits)


def log(s, nl=True):
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = '/gpu:0' if args.cuda else 'CPU'


def run(benchmark_step):
    # Warm-up
    log('Running warmup...')
    timeit.timeit(benchmark_step, number=args.num_warmup_batches)

    # Benchmark
    log('Running benchmark...')
    img_secs = []
    for x in range(args.num_iters):
        time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
        img_secs.append(img_sec)

    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))


loss = loss_function()
train_opt = opt.minimize(loss)

if tf.executing_eagerly():
    with tf.device(device):
        run(lambda: opt.minimize(loss_function,
                                 var_list=model.trainable_variables))
else:
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as session:
        session.run(init)
        if args.kf_optimizer:
            from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
            session.run(BroadcastGlobalVariablesOp())
        run(lambda: session.run(train_opt))
        if barrier_op is not None:
            session.run(barrier_op)
