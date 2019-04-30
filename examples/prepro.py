#!/usr/bin/env python3
import argparse
import os

import tensorflow as tf
from kungfu.helpers import imagenet

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

args = parser.parse_args()
data_dir = args.data_dir

images, labels = imagenet.create_dataset(args.data_dir, args.batch_size)

with tf.Session() as sess:
    for i in range(2):
        xs, y_s = sess.run([images, labels])
        print(xs.shape)
        print(y_s.shape)
