#!/usr/bin/env python3
import os

import tensorflow as tf
from kungfu.helpers import imagenet

data_dir = os.path.join(os.getenv('HOME'), 'Desktop')

batch_size = 32
images, labels = imagenet.create_dataset(data_dir, batch_size)

with tf.Session() as sess:
    for i in range(2):
        xs, y_s = sess.run([images, labels])
        print(xs.shape)
        print(y_s.shape)
