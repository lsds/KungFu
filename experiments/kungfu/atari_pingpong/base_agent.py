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


def cumulative_discounted_reward(rewards, gamma=0.99):
    cdr = []
    c = 0
    for r in rewards:
        if r != 0:
            c = r
        else:
            c = c * gamma + r
        cdr.append(c)
    return normalize(cdr)


class BaseAgent(object):
    def __init__(self, image_shape, actions):
        self._use_kungfu = True

        self.xs = []
        self.ys = []
        self.rs = []

        self._all_vars = []
        self._model_ops = self._build_model_ops(image_shape)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_model_ops(self, image_shape):
        images, probs = self._model(image_shape)
        sampling_prob = tf.nn.softmax(probs)

        actions = tf.placeholder(tf.int32, shape=(None, ))
        discount_rewards = tf.placeholder(tf.float32, shape=(None, ))

        loss = loss_func(probs, actions, discount_rewards)

        learning_rate = 1e-3
        decay_rate = 0.99
        # optmizer = tf.train.GradientDescentOptimizer(learning_rate)
        # optmizer = tf.train.AdamOptimizer(learning_rate)
        optmizer = tf.train.RMSPropOptimizer(learning_rate, decay_rate)

        if self._use_kungfu:
            import kungfu as kf
            optmizer = kf.SyncSGDOptimizer(optmizer)

        train_op = optmizer.minimize(loss)
        return (
            images,
            sampling_prob,
            actions,
            discount_rewards,
            train_op,
        )

    def _model(self, image_shape):
        raise RuntimeError('Not Implemented')

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
                discount_rewards: cumulative_discounted_reward(self.rs, gamma),
            })

        self.xs = []
        self.ys = []
        self.rs = []

        self.save('pingpong.latest.npz')
