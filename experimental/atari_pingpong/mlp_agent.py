from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf

from base_agent import BaseAgent


class Agent(BaseAgent):
    def _new_dense_weight(self, shape):
        # TODO: use deterministic random for test
        initial = tf.truncated_normal(shape, stddev=0.1)
        # initial = tf.zeros(shape) # This doesn't converge
        v = tf.Variable(initial)
        self._all_vars.append(v)
        return v

    def _dense(self, x, n, act=tf.nn.relu):
        input_size = int(x.shape[-1])
        w = self._new_dense_weight((input_size, n))
        y = tf.matmul(x, w)
        if act:
            y = act(y)
        return y

    def _model(self, image_shape):
        image_size = reduce(mul, image_shape, 1)
        x = tf.placeholder(tf.float32, shape=(None, ) + image_shape)
        x_flat = tf.reshape(x, [-1, image_size])
        l1 = self._dense(x_flat, 200)
        l2 = self._dense(l1, 3, act=None)
        return x, l2
