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
from tensorflow.python.util import deprecation
from kungfu.python import (current_rank, 
    log_stats, print_strategy_stats,
    check_interference, current_cluster_size)
from kungfu.tensorflow.ops.adapt import set_tree, calc_stats

deprecation._PRINT_DEPRECATION_WARNINGS = False
default_strategy = 0
alternative_strategy = 1

def get_alternative_star_strategy(off: int):
    tree = [off] * current_cluster_size()
    return tree

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
parser.add_argument('--adapt',
                    action='store_true',
                    default=False,
                    help='Strategy Adaptation')

args = parser.parse_args()
args.cuda = not args.no_cuda

config = tf.ConfigProto()
if args.cuda:
    config.gpu_options.allow_growth = True
    from kungfu.python import _get_cuda_index
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
    elif args.kf_optimizer == 'sync-sgd-monitor':
        from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
        opt = SynchronousSGDOptimizer(opt, monitor=True)
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
    from kungfu.tensorflow.ops import current_rank
    if current_rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = '/gpu:0' if args.cuda else 'CPU'


def log_detailed_result(value, error, attrs):
    import json
    attr_str = json.dumps(attrs, separators=(',', ':'))
    # grep -o RESULT.* *.log
    print('RESULT: %f +-%f %s' % (value, error, attr_str))


def log_final_result(value, error):
    from kungfu.tensorflow.ops import current_rank, current_cluster_size
    if current_rank() != 0:
        return
    attrs = {
        'framework': 'kungfu',
        'np': current_cluster_size(),
        'strategy': os.getenv('KUNGFU_ALLREDUCE_STRATEGY'),
        'bs': args.batch_size,
        'model': args.model,
        'kf-opt': args.kf_optimizer,
        'fuse': args.fuse,
    }
    log_detailed_result(value, error, attrs)


def run(benchmark_step, session):
    # Warm-up
    log('Running warmup...')
    for x in range(args.num_warmup_batches):
        time = timeit.timeit(benchmark_step, number=1)
        img_sec = args.batch_size / time
        log('Warmup Step #%d: %.1f img/sec per %s, took %.3fs' %
            (x, img_sec, device, time))

    set_tree_op = set_tree(tree=get_alternative_star_strategy(alternative_strategy))
    calc_stats_op = calc_stats()

    changed = False
    # Benchmark
    log('Running benchmark...')
    img_secs = []
    for x in range(args.num_iters): 
        for y in range(args.num_batches_per_iter):
            time = timeit.timeit(benchmark_step, number=1)
            session.run(calc_stats_op)
            log_stats()

            img_sec = args.batch_size / time
            log('Iter #%d: %.1f img/sec per %s' % (x*args.num_batches_per_iter + y, img_sec, device))
            img_secs.append(img_sec)

            if args.adapt and (x*args.num_iters + y > 4):
                if changed:
                    continue
                ret = check_interference()
                log('Cluster response ', ret)
                if ret == 1:
                    changed = True
                    log('Interference detected. Changing to alternative comm strategy !')
                    session.run(set_tree_op)
    
    if current_rank() == 0:
        print_strategy_stats()

    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
    log_final_result(img_sec_mean, img_sec_conf)


loss = loss_function()
train_opt = opt.minimize(loss)

if tf.executing_eagerly():
    with tf.device(device):
        run(lambda: opt.minimize(loss_function,
                                 var_list=model.trainable_variables))
else:
    init = tf.global_variables_initializer()
    bcast_op = None
    if args.kf_optimizer:
        from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
        bcast_op = BroadcastGlobalVariablesOp()
    with tf.Session(config=config) as session:
        from kungfu._utils import measure
        duration, _ = measure(lambda: session.run(init))
        log('init took %.3fs' % (duration))
        if bcast_op:
            duration, _ = measure(lambda: session.run(bcast_op))
            log('bcast_op took %.3fs' % (duration))
        run(lambda: session.run(train_opt), session)
        if barrier_op is not None:
            session.run(barrier_op)
