import argparse
import os

import kungfu.tensorflow as kf
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
    p.add_argument('--model', type=str, default='ResNet50')
    p.add_argument('--epoch-size', type=int, default=1024)
    p.add_argument('--epoch-num', type=int, default=1)
    p.add_argument('--max-train-steps', type=int, default=100)
    p.add_argument('--init-batch-size', type=int, default=32)
    p.add_argument('--model-dir', type=str, default='ckpt')
    return p.parse_args()


image_shape = {
    'ImageNet': [224, 224, 3],
    'CIFAR10': [32, 32, 3],
}


def random_input(sample_shape=None, feature_shape=None):
    if sample_shape is None:
        sample_shape = image_shape['CIFAR10']
    if feature_shape is None:
        feature_shape = []
    samples = tf.random_uniform(sample_shape)
    labels = tf.random_uniform(feature_shape,
                               minval=0,
                               maxval=999,
                               dtype=tf.int64)
    return samples, labels


def batched_random_input(batch_size, sample_shape=None, feature_shape=None):
    if sample_shape is None:
        sample_shape = image_shape['CIFAR10']
    if feature_shape is None:
        feature_shape = []
    samples = tf.random_uniform([batch_size] + sample_shape)
    labels = tf.random_uniform([batch_size] + feature_shape,
                               minval=0,
                               maxval=999,
                               dtype=tf.int64)

    return samples, labels


def random_dataset(epoch_size, init_batch_size=32):
    samples, labels = random_input()
    samples = tf.data.Dataset.from_tensors(samples)
    labels = tf.data.Dataset.from_tensors(labels)
    features = {'x': samples}
    ds = tf.data.Dataset.zip((features, labels))
    ds = ds.repeat(epoch_size)
    batch_size = tf.Variable(init_batch_size,
                             dtype=tf.int64,
                             name='batch_size',
                             trainable=False)
    ds = ds.batch(batch_size)
    return ds, batch_size


def build_input_fn(init_batch_size, steps=None):
    def input_fn():
        batch_size = kf.get_or_create_batch_size_tensor(init_batch_size)
        samples, labels = batched_random_input(batch_size)
        samples = tf.data.Dataset.from_tensors(samples)
        labels = tf.data.Dataset.from_tensors(labels)
        features = {'x': samples}
        ds = tf.data.Dataset.zip((features, labels))
        ds = ds.repeat(steps)
        return ds

    return input_fn


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


def train_one_epoch(sess, epoch, init_dataset_op, sync_op, train_op, policy,
                    variables, params, max_step, size_place, resize_op):
    np = current_cluster_size()
    policy.before_epoch(variables, params)
    grads = []
    sess.run(init_dataset_op)
    step = 0
    need_sync = True
    gs = tf.train.get_or_create_global_step()
    while True:
        if need_sync:
            if detached():
                break
            sess.run(sync_op)
            need_sync = False
        try:
            sess.run(train_op)
            step += 1
        except tf.errors.OutOfRangeError:
            break

        # v_gns = sess.run(gns)
        policy.after_step(variables, params, grads)
        step = sess.run(gs)
        # print('trained %d steps' % (step))
        if step >= max_step:
            break
        # print('trained %d steps, gns: %f' % (v_step, v_gns))
        need_sync = sess.run(resize_op, feed_dict={size_place: np})
    policy.after_epoch(variables, params)
    return step


def train(init_op, init_dataset_op, sync_op, train_op, policy, epoch_num,
          max_step):
    np = current_cluster_size()
    size_place = tf.placeholder(dtype=tf.uint32)
    resize_op = resize(size_place)

    variables = []
    params = []
    grads = []

    with tf.Session() as sess:
        policy.before_train(variables, params)
        sess.run(init_op)
        for epoch in range(epoch_num):
            train_one_epoch(sess, epoch, init_dataset_op, sync_op, train_op,
                            policy, variables, params, max_step, size_place,
                            resize_op)
            if detached():
                break
        print('finished')


def train_with_simple_session(args):
    policy = GNSPolicy()
    ds, batch_size_tensor = random_dataset(args.epoch_size, args.batch_size)
    it = ds.make_initializable_iterator()
    features, labels = it.get_next()
    train_op, _loss = build_train_op(args.model, features, labels)
    sync_op = BroadcastGlobalVariablesOp()
    init_op = tf.global_variables_initializer()
    train(init_op, it.initializer, sync_op, train_op, policy, args.epoch_num,
          args.max_train_steps)


def train_with_estimator(args):
    print('BEFORE build_estimator')
    classifier = build_estimator(args)
    print('AFTER build_estimator')

    print('BEFORE build_input_fn')
    input_fn = build_input_fn(args.init_batch_size)
    print('AFTER build_input_fn')

    from gns_policy import GNSPolicy
    policy = GNSPolicy(args.epoch_size, args.epoch_num, args.init_batch_size)
    hooks = [policy.get_tensorflow_hook()]

    print('BEFORE train')
    classifier.train(input_fn, hooks=hooks)
    print('AFTER train')


def main():
    args = parse_args()
    # train_with_simple_session(args)
    train_with_estimator(args)


main()
