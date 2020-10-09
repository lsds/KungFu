import argparse
import os

import kungfu.tensorflow as kf
import tensorflow as tf
from kungfu.python import current_cluster_size, detached
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import resize
from kungfu.tensorflow.optimizers import MonitorGradientNoiseScaleOptimizer
from tensorflow.python.util import deprecation

from gns_policy import GNSPolicy
from input_data import build_cifar10_input_fn, build_random_input_fn

deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_args():
    p = argparse.ArgumentParser(
        description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', type=str, default='ResNet50')
    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--epoch-size', type=int, default=1024)
    p.add_argument('--epoch-num', type=int, default=1)
    p.add_argument('--max-train-steps', type=int, default=100)
    p.add_argument('--init-batch-size', type=int, default=32)
    p.add_argument('--model-dir', type=str, default='ckpt')
    p.add_argument('--fake-data', action='store_true', default=False)
    return p.parse_args()


def build_optimizer(learning_rate=0.01, init_device_batch_size=32):
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    device_batch_size = tf.Variable(init_device_batch_size,
                                    dtype=tf.int32,
                                    trainable=False,
                                    name='device_batch_size')
    opt = MonitorGradientNoiseScaleOptimizer(opt,
                                             device_batch_size,
                                             verbose=False)
    return opt


def get_model(name):  # name = 'ResNet50'
    from tensorflow.keras import applications
    return getattr(applications, name)(weights=None)


def build_train_op(model_name, features, labels):
    model = get_model(model_name)

    logits = model(features['x'], training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    opt = build_optimizer()
    gs = tf.train.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=gs)
    return train_op, loss


def build_model_fn(model_name):
    def model_fn(features, labels, mode):
        train_op, loss = build_train_op(model_name, features, labels)
        eval_metric_ops = None
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          eval_metric_ops=eval_metric_ops)

    return model_fn


def _get_model_dir(model_dir):
    from kungfu.python import uid
    x = uid()
    port = (x >> 16) & 0xffff
    version = x & 0xffff
    suffix = '%d.%d' % (port, version)
    return os.path.join(model_dir, suffix)


def build_estimator(args):
    model_dir = _get_model_dir(args.model_dir)
    model_fn = build_model_fn(args.model)
    classifier = tf.estimator.Estimator(model_fn, model_dir=model_dir)
    return classifier


def train_with_estimator(args):
    print('BEFORE build_estimator')
    classifier = build_estimator(args)
    print('AFTER build_estimator')

    print('BEFORE build_input_fn')
    if args.fake_data:
        input_fn = build_random_input_fn(args.init_batch_size)
    else:
        input_fn = build_cifar10_input_fn(args.init_batch_size)
    print('AFTER build_input_fn')

    policy = GNSPolicy(args.epoch_size, args.epoch_num, args.init_batch_size)
    hooks = [policy.get_tensorflow_hook()]

    print('BEFORE train')
    classifier.train(input_fn, hooks=hooks)
    print('AFTER train')


def main():
    args = parse_args()
    train_with_estimator(args)


main()
