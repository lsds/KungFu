from operator import mul
from functools import reduce

import numpy as np
import tensorflow as tf

from base_agent import BaseAgent, loss_func


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def expand_shape(shape):
    dims = []
    for s in shape:
        d = None
        try:
            d = int(s)
        except:
            pass
        dims.append(d)
    return dims


class Agent(BaseAgent):
    def _new_weight(self, shape):
        # TODO: use deterministic random for test
        initial = tf.truncated_normal(shape, stddev=0.1)
        # initial = tf.zeros(shape) # This doesn't converge
        v = tf.Variable(initial)
        self._all_vars.append(v)
        return v

    def _dense(self, x, n, act=tf.nn.relu):
        _n, _h, _w, n_channels = expand_shape(x.shape)
        image_size = _h * _w * n_channels
        x_flat = tf.reshape(x, [-1, image_size])
        w = self._new_weight((image_size, n))
        y = tf.matmul(x_flat, w)
        if act:
            y = act(y)
        return y

    # FIXME: implement
    def _conv(self, x, n_fiters, filter_size, act=tf.nn.relu):
        _n, _h, _w, n_channels = expand_shape(x.shape)
        r, s = filter_size
        w = self._new_weight((r, s, n_channels, n_fiters))
        y = tf.nn.conv2d(x, w, strides=(1, 2, 2, 1), padding='SAME')
        if act:
            y = act(y)
        return y

    def _cnn(self, image_shape):
        x = tf.placeholder(tf.float32, shape=(None, ) + image_shape)
        x_vol = tf.reshape(x, (-1, ) + image_shape + (1, ))
        l1 = self._conv(x_vol, 16, (3, 3))
        l2 = self._dense(l1, 3, act=None)
        return x, l2

    def _model(self, image_shape):
        images, probs = self._cnn(image_shape)
        sampling_prob = tf.nn.softmax(probs)

        actions = tf.placeholder(tf.int32, shape=(None, ))
        discount_rewards = tf.placeholder(tf.float32, shape=(None, ))

        loss = loss_func(probs, actions, discount_rewards)

        learning_rate = 1e-3
        # decay_rate = 0.99
        # optmizer = tf.train.RMSPropOptimizer(learning_rate, decay_rate)
        optmizer = tf.train.GradientDescentOptimizer(learning_rate)
        # use_kungfu = False
        # if use_kungfu:
        #     import kungfu as kf
        #     optmizer = kf.SyncSGDOptimizer(optmizer)
        train_op = optmizer.minimize(loss)
        return (
            images,
            sampling_prob,
            actions,
            discount_rewards,
            train_op,
        )
