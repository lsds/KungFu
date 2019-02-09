from operator import mul
from functools import reduce

import numpy as np
import tensorflow as tf


def loss_func(logits, actions, rewards):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=logits)
    return tf.reduce_sum(tf.multiply(cross_entropy, rewards))


def normalize(x):
    x -= np.mean(x)
    std = np.std(x)
    if std != 0:
        x /= std
    return x


def discount_episode_rewards(rewards, gamma=0.99):
    discounted_r = [0 for _ in rewards]
    running_add = 0
    for t in reversed(range(len(rewards))):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return normalize(discounted_r)


class Agent(object):
    def __init__(self, init, actions):
        tf.reset_default_graph()

        self.xs = []
        self.ys = []
        self.rs = []

        self._all_vars = []
        self._model_ops = self._model(init)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def save(self, filename):
        npz = dict()
        values = self.sess.run(self._all_vars)
        for v, var in zip(values, self._all_vars):
            npz[var.name] = v
        np.savez(filename, **npz)

    def load(self, filename):
        npz = np.load(filename)

        all_vars = dict()
        for v in self._all_vars:
            all_vars[v.name] = v

        ops = []
        for name, value in npz.items():
            ops.append(tf.assign(all_vars[name], value))

        self.sess.run(ops)

    def act(self, image):
        (
            images,
            sampling_prob,
            actions,
            discount_rewards,
            train_op,
        ) = self._model_ops
        prob = self.sess.run(sampling_prob, feed_dict={images: [image]})
        return np.random.choice(np.arange(len(prob[0])), p=prob[0])

    def percept(self, image, action, reward):
        self.xs.append(image)
        self.ys.append(action)
        self.rs.append(reward)

    def learn(self):
        (
            images,
            sampling_prob,
            actions,
            discount_rewards,
            train_op,
        ) = self._model_ops

        gamma = 0.99
        self.sess.run(
            train_op,
            feed_dict={
                images: self.xs,
                actions: self.ys,
                discount_rewards: discount_episode_rewards(self.rs, gamma),
            })

        self.xs = []
        self.ys = []
        self.rs = []

        self.save('pingpong.latest.npz')

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

    def _mlp(self, image_shape):
        image_size = reduce(mul, image_shape, 1)
        x = tf.placeholder(tf.float32, shape=(None, ) + image_shape)
        x_flat = tf.reshape(x, [-1, image_size])
        l1 = self._dense(x_flat, 200)
        l2 = self._dense(l1, 3, act=None)
        return x, l2

    def _model(self, image_shape):
        images, probs = self._mlp(image_shape)
        sampling_prob = tf.nn.softmax(probs)

        actions = tf.placeholder(tf.int32, shape=(None, ))
        discount_rewards = tf.placeholder(tf.float32, shape=(None, ))

        loss = loss_func(probs, actions, discount_rewards)

        learning_rate = 1e-3
        decay_rate = 0.99
        optmizer = tf.train.RMSPropOptimizer(learning_rate, decay_rate)
        # optmizer = tf.train.GradientDescentOptimizer(learning_rate)
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
