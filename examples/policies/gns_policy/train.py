import argparse

import tensorflow as tf
from kungfu.python import current_cluster_size, detached
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import resize
from kungfu.tensorflow.optimizers import MonitorGradientNoiseScaleOptimizer
from kungfu.tensorflow.optimizers.grad_noise_scale import get_gns_tensor
from tensorflow.python.util import deprecation

from gns_policy import GNSPolicy

deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_args():
    p = argparse.ArgumentParser(
        description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', type=str, default='ResNet50', help='model')
    p.add_argument('--batch-size', type=int, default=32, help='batch size')
    p.add_argument('--train-steps', type=int, default=100, help='')
    return p.parse_args()


def fake_input(batch_size):
    samples = tf.random_uniform([batch_size, 224, 224, 3])
    labels = tf.random_uniform([batch_size, 1],
                               minval=0,
                               maxval=999,
                               dtype=tf.int64)
    return samples, labels


def build_optimizer(learning_rate=0.01, init_device_batch_size=32):
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    device_batch_size = tf.Variable(init_device_batch_size,
                                    dtype=tf.int32,
                                    trainable=False,
                                    name='device_batch_size')
    opt = MonitorGradientNoiseScaleOptimizer(opt, device_batch_size)
    return opt


def get_model(name):
    from tensorflow.keras import applications
    return getattr(applications, name)(weights=None)


def build_train_op(model_name, features, labels):
    model = get_model(model_name)

    logits = model(features['x'], training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    opt = build_optimizer()
    train_op = opt.minimize(loss)

    return train_op


def train_with_simple_session(args):
    policy = GNSPolicy()
    np = current_cluster_size()
    samples, labels = fake_input(args.batch_size)
    features = {'x': samples}
    gs = tf.train.get_or_create_global_step()
    sync_op = BroadcastGlobalVariablesOp()
    train_op = build_train_op(args.model, features, labels)
    gns = get_gns_tensor()
    size_place = tf.placeholder(dtype=tf.uint32)
    resize_op = resize(size_place)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        need_sync = True
        for step in range(1, 1 + args.train_steps):
            if need_sync:
                if detached():
                    break
                sess.run(sync_op)
                need_sync = False
            sess.run(train_op)
            v_gns = sess.run(gns)
            v_step = sess.run(gs)
            print('trained %d steps, gns: %f' % (v_step, v_gns))

            need_sync = sess.run(resize_op, feed_dict={size_place: np})


def main():
    args = parse_args()
    train_with_simple_session(args)


main()
