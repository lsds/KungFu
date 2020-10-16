#!/usr/bin/env python3
import argparse
import glob
import os
import time

import tensorflow as tf
from kungfu.datasets.adaptor import ExampleDatasetAdaptor
from kungfu.tensorflow.v1.helpers import imagenet

parser = argparse.ArgumentParser(
    description='imagenet dataset example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data-dir',
                    type=str,
                    default=os.path.join(os.getenv('HOME'), 'Desktop'),
                    help='dir to dataset')
parser.add_argument('--batch-size',
                    type=int,
                    default=32,
                    help='input batch size')
parser.add_argument('--pattern',
                    type=str,
                    default='train-*-of-*',
                    help='file name pattern')
parser.add_argument('--adaptive',
                    type=bool,
                    default=False,
                    help='enable adaptive training')


def adaptive_train(original_ds, adapt, handle):
    init_ds, get_next = adapt(original_ds)

    update_offset_op = adapt.create_update_offset()
    update_topology_op = adapt.create_update_topology()
    rewind_op = adapt.create_rewind()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        adapt.debug(sess)

        epoches = 10
        for epoch in range(epoches):
            print('BEGIN epoch')
            print('epoch %d' % epoch)
            sess.run(init_ds)
            adapt.debug(sess)

            step = 0
            while True:
                step += 1
                try:
                    v = sess.run(get_next)
                    print('epoch: %d, step: %d' % (epoch, step))
                    handle(v)
                    sess.run(update_offset_op)
                    adapt.debug(sess)
                except tf.errors.OutOfRangeError:
                    print('[W] rewind to beginning!')
                    sess.run(rewind_op)
                    sess.run(init_ds)
                    adapt.debug(sess)
                if step > 4:
                    break
            sess.run(update_topology_op)
            print('END epoch')
            print('\n')


def simple_train(original_ds, batch_size, handle):
    ds = original_ds.batch(batch_size)
    it = ds.make_one_shot_iterator()
    get_next = it.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            try:
                v = sess.run(get_next)
                step += 1
                handle(v)
            except tf.errors.OutOfRangeError:
                print('stopped after %s steps' % (step))
                break


class Reporter(object):
    def __init__(self):
        self._t0 = time.time()
        self._last = self._t0
        self._done = 0

    def report(self, n):
        t1 = time.time()
        d = t1 - self._last
        self._last = t1
        self._done += n
        rate = float(n) / float(d)
        return '%.2f images/s, %.2fs past, ' % (rate, t1 - self._t0)


def main():
    args = parser.parse_args()
    full_pattern = os.path.join(args.data_dir, args.pattern)
    filenames = glob.glob(full_pattern)
    if len(filenames) == 0:
        raise RuntimeError('no data files found')

    ds = tf.data.TFRecordDataset(filenames)
    ds = ds.map(imagenet.record_to_labeled_image)

    r = Reporter()

    def handle(v):
        xs, y_s = v
        bs = int(xs.shape[0])
        print('xs :: %s, y_s :: %s, %s' % (xs.shape, y_s.shape, r.report(bs)))

    if args.adaptive:
        shard_count = 1  # FIXME: get from ENV
        adapt = ExampleDatasetAdaptor(batch_size=args.batch_size,
                                      shard_count=shard_count)
        adaptive_train(ds, adapt, handle)
    else:
        simple_train(ds, args.batch_size, handle)


main()
