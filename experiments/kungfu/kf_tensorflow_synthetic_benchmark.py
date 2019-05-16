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
parser.add_argument('--no-kungfu',
                    action='store_true',
                    default=False,
                    help='disables kungfu')
# FIXME: rename this file
parser.add_argument('--dataset',
                    type=str,
                    default='synthetic',
                    help='synthetic | imagenet')
parser.add_argument('--data-dir', type=str, default='', help='dir to dataset')
parser.add_argument('--data-records',
                    type=int,
                    default=1024,
                    help='number of TFRecord files')

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

# Set up standard model.
model = getattr(applications, args.model)(weights=None)

opt = tf.train.GradientDescentOptimizer(0.01)

# Kungfu: wrap optimizer with ParallelOptimizer.
if args.kungfu:
    from kungfu.optimizers import ParallelOptimizer
    opt = ParallelOptimizer(opt, device_batch_size=args.batch_size)
 
init = tf.global_variables_initializer()

if args.dataset == 'imagenet':
    from kungfu.helpers import imagenet
    data, target = imagenet.create_dataset(args.data_dir, args.batch_size,
                                           args.data_records)
else:
    data = tf.random_uniform([args.batch_size, 224, 224, 3])
    target = tf.random_uniform([args.batch_size, 1],
                               minval=0,
                               maxval=999,
                               dtype=tf.int64)


def loss_and_accuracy():
    logits = model(data, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(target, logits)
    correct_prediction = tf.equal(target, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss, accuracy


def loss_function():
    loss, _ = loss_and_accuracy()
    return loss


def log(s, nl=True):
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = '/gpu:0' if args.cuda else 'CPU'


def run(benchmark_step, eval_step=None):
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
        if eval_step:
            loss, acc = eval_step()
            print('loss: %f,accuracy: %s' % (loss, acc))

    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))


if tf.executing_eagerly():
    with tf.device(device):
        run(lambda: opt.minimize(loss_function,
                                 var_list=model.trainable_variables))
else:
    with tf.Session(config=config) as session:
        init.run()

        loss, acc = loss_and_accuracy()
        train_opt = opt.minimize(loss)
        run(lambda: session.run(train_opt), lambda: session.run([loss, acc]))
