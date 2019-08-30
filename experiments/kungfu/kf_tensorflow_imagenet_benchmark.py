#!/usr/bin/env python3
"""
Modified from:
https://github.com/uber/horovod/blob/master/examples/tensorflow_synthetic_benchmark.py
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import os
import numpy as np
import timeit

from kungfu.helpers import imagenet
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
parser.add_argument('--data-dir', type=str, default='', help='dir to dataset')
parser.add_argument('--pattern',
                    type=str,
                    default='train-*-of-*',
                    help='file name pattern')

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


def create_dataset(batch_size):
    full_pattern = os.path.join(args.data_dir, args.pattern)
    filenames = glob.glob(full_pattern)
    if len(filenames) == 0:
        raise RuntimeError('no data files found')
    print('using %d records' % (len(filenames)))
    ds = tf.data.TFRecordDataset(filenames)
    ds = ds.map(imagenet.record_to_labeled_image)
    ds = ds.batch(batch_size)
    it = ds.make_one_shot_iterator()
    get_next = it.get_next()
    return get_next


# Set up standard model.
model = getattr(applications, args.model)(weights=None)

learning_rate = 0.01
# learning_rate = 1
# momentum = 0.9
# opt = tf.train.GradientDescentOptimizer(0.01)
opt = tf.train.AdamOptimizer(learning_rate)
# opt = tf.train.MomentumOptimizer(learning_rate,
#                                  momentum,
#                                  use_nesterov=True)

# Kungfu: wrap optimizer with SyncSGDOptimizer.
if args.kungfu:
    from kungfu.optimizers import SyncSGDOptimizer
    opt = SyncSGDOptimizer(opt)

if args.data_dir:
    print('Using real dataset!')
    data, target = create_dataset(args.batch_size)
else:
    print('Using synthetic dataset!')
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
    # timeit.timeit(benchmark_step, number=args.num_warmup_batches)

    # Benchmark
    log('Running benchmark...')
    # img_secs = []
    for x in range(args.num_iters):
        for y in range(args.num_batches_per_iter):
            _, loss = benchmark_step()
            print('loss: %f' % (loss))
        # time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
        # img_sec = args.batch_size * args.num_batches_per_iter / time
        # log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
        # img_secs.append(img_sec)

    # Results
    # img_sec_mean = np.mean(img_secs)
    # img_sec_conf = 1.96 * np.std(img_secs)
    # log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))


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
        run(lambda: session.run([train_opt, loss]))
