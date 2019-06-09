#!/usr/bin/env python3
import tensorflow as tf
from kungfu.optimizers import PeerModelAveraging


class FakeOptimizer(object):
    def __init__(self, *args, **kwargs):
        pass

    def apply_gradients(self, grads_and_vars, **kwargs):
        return tf.group([tf.assign(v, v - g) for g, v in grads_and_vars])


x = tf.Variable(tf.ones([3, 3, 3]))
init = PeerModelAveraging.get_initializer()

variables = [x]
gradients = [tf.Variable(tf.zeros(x.shape)) for x in variables]

optimizer = FakeOptimizer()
opt = PeerModelAveraging(optimizer)

train_step = opt.apply_gradients(zip(gradients, variables))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(init)

    n_iters = 11
    steps_per_iter = 10

    for _ in range(n_iters):
        for _ in range(steps_per_iter):
            sess.run(train_step)
