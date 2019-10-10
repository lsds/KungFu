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
from kungfu.internal import _get_self_rank


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
parser.add_argument('--kungfu',
                    type=str,
                    default='sync-sgd',
                    help='kungfu optimizer')
parser.add_argument('--kungfu-fuse-variables',
                    type=bool,
                    default=True,
                    help='fuse variables')
parser.add_argument("--ps_hosts",
                    type=str,
                    default="",
                    help="Comma-separated list of hostname:port pairs")
parser.add_argument("--worker_hosts",
                    type=str,
                    default="",
                    help="Comma-separated list of hostname:port pairs")

args = parser.parse_args()
args.cuda = not args.no_cuda



task_index = _get_self_rank() % 2
job_name = "ps" if task_index < 2 else "worker"


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

def main(_):
    global task_index, job_name, train_opt
    ps_hosts = args.ps_hosts.split(",")
    worker_hosts = args.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                            job_name=job_name,
                            task_index=task_index)

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % task_index,
            cluster=cluster)):

            # Set up standard model.
            model = getattr(applications, args.model)(weights=None)

            opt = tf.train.GradientDescentOptimizer(0.01)

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

            loss = loss_function()
            train_opt = opt.minimize(loss)

            
        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        print("Task index " + str(task_index))
        with tf.train.MonitoredTrainingSession(master=server.target,
                                            is_chief=(task_index == 0),
                                            checkpoint_dir="/tmp/train_logs") as mon_sess:
            # init = tf.global_variables_initializer()                                        
            # mon_sess.run(init)
            run(lambda: mon_sess.run(train_opt))


if __name__ == "__main__":
  tf.app.run(main=main)