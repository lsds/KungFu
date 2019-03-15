#!/usr/bin/env python3
"""
Modified from:
https://github.com/uber/horovod/blob/master/examples/tensorflow_synthetic_benchmark.py
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import numpy as np
import timeit

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import Input

# Benchmark settings
parser = argparse.ArgumentParser(
    description='TensorFlow Synthetic Benchmark',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--model', type=str, default='ResNet50', help='model to benchmark')
parser.add_argument(
    '--image-format',
    type=str,
    default='channels_last',
    help='channels_last | channels_first')
parser.add_argument(
    '--batch-size', type=int, default=32, help='input batch size')

parser.add_argument(
    '--num-warmup-batches',
    type=int,
    default=10,
    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument(
    '--num-batches-per-iter',
    type=int,
    default=10,
    help='number of batches per benchmark iteration')
parser.add_argument(
    '--num-iters', type=int, default=10, help='number of benchmark iterations')

parser.add_argument(
    '--eager',
    action='store_true',
    default=False,
    help='enables eager execution')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--no-kungfu', action='store_true', default=False, help='disables kungfu')

args = parser.parse_args()
args.cuda = not args.no_cuda
args.kungfu = not args.no_kungfu

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
if args.cuda:
    config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = str(hvd.local_rank())
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config.gpu_options.allow_growth = False
    config.gpu_options.visible_device_list = ''

if args.eager:
    tf.enable_eager_execution(config)

if args.image_format == 'channels_first':
    input_shape = (3, 244, 224)
elif args.image_format == 'channels_last':
    input_shape = (244, 224, 3)
else:
    raise RuntimeError('invalid image_format: %s' % args.image_format)

input_tensor = Input(shape=input_shape)

# Set up standard model.
model = getattr(applications, args.model)(
    input_tensor=input_tensor, weights=None)

opt = tf.train.GradientDescentOptimizer(0.01)

# Kungfu: wrap optimizer with ParallelOptimizer.
if args.kungfu:
    from kungfu.optimizers import ParallelOptimizer
    opt = ParallelOptimizer(opt)

init = tf.global_variables_initializer()

data = tf.random_uniform((args.batch_size, ) + input_shape)
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


if tf.executing_eagerly():
    with tf.device(device):
        run(lambda: opt.minimize(
            loss_function, var_list=model.trainable_variables))
else:
    with tf.Session(config=config) as session:
        init.run()

        loss = loss_function()
        train_opt = opt.minimize(loss)
        run(lambda: session.run(train_opt))
