#!/usr/bin/env python3
"""
https://github.com/uber/horovod/blob/master/examples/tensorflow_synthetic_benchmark.py

Please refer to Horovod page to see how to run this script.

$ horovodrun -np 4 python3 benchmark_horovod_torch.py

"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import os
import timeit

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

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
parser.add_argument('--xla',
                    action='store_true',
                    default=False,
                    help='enable XLA')
parser.add_argument('--data-dir', type=str, default='', help='dir to dataset')
parser.add_argument('--file-pattern', type=str, default='train-*-of-*')

args = parser.parse_args()
args.cuda = not args.no_cuda

hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
if args.cuda:
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config.gpu_options.allow_growth = False
    config.gpu_options.visible_device_list = ''

if args.xla:
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

if args.eager:
    tf.enable_eager_execution(config)

# Set up standard model.
model = getattr(applications, args.model)(weights=None)

opt = tf.train.GradientDescentOptimizer(0.01)

# Horovod: wrap optimizer with DistributedOptimizer.
# To make a fair comparison with KungFu, we configure Horovod to use CPUs to run MPI and gradient averaging.
opt = hvd.DistributedOptimizer(opt)

init = tf.global_variables_initializer()
bcast_op = hvd.broadcast_global_variables(0)


def random_input():
    data = tf.random_uniform([args.batch_size, 224, 224, 3])
    target = tf.random_uniform([args.batch_size, 1],
                               minval=0,
                               maxval=999,
                               dtype=tf.int64)
    return data, target


def disk_input(data_dir):
    from kungfu.tensorflow.v1.helpers import imagenet
    filenames = glob.glob(os.path.join(data_dir, args.file_pattern))
    filenames *= 100  # make it long enough
    return imagenet.create_dataset_from_files(filenames, args.batch_size)


def loss_function():
    if args.data_dir:
        data, target = disk_input(args.data_dir)
    else:
        data, target = random_input()
    logits = model(data, training=True)
    return tf.losses.sparse_softmax_cross_entropy(target, logits)


def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))


def log_detailed_result(value, error, attrs):
    import json
    attr_str = json.dumps(attrs, separators=(',', ':'))
    # grep -o RESULT.* *.log
    print('RESULT: %f +-%f %s' % (value, error, attr_str))


def log_final_result(value, error):
    if hvd.rank() > 0:
        return
    import horovod
    attrs = {
        'framework': 'horovod',
        'version': horovod.__version__,
        'np': hvd.size(),
        'bs': args.batch_size,
        'model': args.model,
        'xla': args.xla,
        'data': 'disk' if args.data_dir else 'memory',
    }
    try:
        attrs['nccl_built'] = hvd.nccl_built()
    except:
        pass
    log_detailed_result(value, error, attrs)


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
    log('Total img/sec on %d %s(s): %.1f +-%.1f' %
        (hvd.size(), device, hvd.size() * img_sec_mean,
         hvd.size() * img_sec_conf))
    log_final_result(img_sec_mean, img_sec_conf)


if tf.executing_eagerly():
    with tf.device(device):
        run(lambda: opt.minimize(loss_function,
                                 var_list=model.trainable_variables))
else:
    with tf.Session(config=config) as session:
        init.run()
        bcast_op.run()

        loss = loss_function()
        train_opt = opt.minimize(loss)
        run(lambda: session.run(train_opt))
