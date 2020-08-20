import argparse
import os

from kungfu._utils import _log_event, one_based_range

import debug_hooks
import tensorflow as tf
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_args():
    p = argparse.ArgumentParser(
        description='TensorFlow Synthetic Benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model',
                   type=str,
                   default='ResNet50',
                   help='model to benchmark')
    p.add_argument('--elastic', action='store_true', default=False, help='')
    p.add_argument('--batch-size', type=int, default=32, help='batch size')
    p.add_argument('--train-steps', type=int, default=10, help='')
    p.add_argument('--epochs', type=int, default=10, help='')
    p.add_argument('--epoch-size', type=int, default=10, help='')
    p.add_argument('--sync-step', action='store_true', default=False, help='')
    p.add_argument('--resize-schedule',
                   type=str,
                   default='10:2,100:0',
                   help='')
    p.add_argument('--model-dir', type=str, default='ckpt', help='')
    p.add_argument('--tf-method',
                   type=str,
                   default='simple',
                   help='simple | monitored | estimator')
    p.add_argument('--show-training-throughput',
                   action='store_true',
                   default=False,
                   help='')
    return p.parse_args()


def build_optimizer(learning_rate=0.01):
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
    opt = SynchronousSGDOptimizer(opt)
    return opt


def simple_input(batch_size):
    samples = tf.random_uniform([batch_size, 224, 224, 3])
    labels = tf.random_uniform([batch_size, 1],
                               minval=0,
                               maxval=999,
                               dtype=tf.int64)

    return samples, labels


def build_input_fn(batch_size, steps=None):
    def input_fn():
        samples, labels = simple_input(batch_size)
        samples = tf.data.Dataset.from_tensors(samples)
        labels = tf.data.Dataset.from_tensors(labels)
        features = {'x': samples}
        ds = tf.data.Dataset.zip((features, labels))
        ds = ds.repeat(steps)
        return ds

    return input_fn


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


def build_model_fn(model_name):
    def model_fn(features, labels, mode):
        model = get_model(model_name)

        logits = model(features['x'], training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        opt = build_optimizer()
        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())

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


def run_simple_session(args):
    samples, labels = simple_input(args.batch_size)
    features = {'x': samples}
    train_op = build_train_op(args.model, features, labels)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in one_based_range(args.train_steps):
            sess.run(train_op)
            print('trained %d steps' % (step))


def run_with_session_and_hooks(args):
    # samples, labels = simple_input(args.batch_size) # FIXME: infinite loop
    # features = {'x': samples}

    input_fn = build_input_fn(args.batch_size, args.train_steps)
    ds = input_fn()
    it = ds.make_initializable_iterator()
    features, labels = it.get_next()

    train_op = build_train_op(args.model, features, labels)

    hooks = [
        # debug_hooks.LogStepHook(),
    ]

    if args.show_training_throughput:
        hooks.append(debug_hooks.LogPerfHook(args.batch_size))

    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        sess.run(it.initializer)  # FIXME: don't init in hooks
        step = 0
        while sess.should_stop:
            sess.run(train_op)
            step += 1

    print('MonitoredTrainingSession trained %d steps' % (step))


def parse_scheule(schedule):
    d = dict()
    for kv in schedule.split(','):
        k, v = kv.split(':')
        d[int(k)] = int(v)
    return d


def run_with_estimator(args):
    _log_event('BEGIN :: run_with_estimator')

    _log_event('BEGIN :: build_estimator')
    classifier = build_estimator(args)
    _log_event('END :: build_estimator')

    hooks = [
        debug_hooks.LogStepHook(),
    ]

    if args.show_training_throughput:
        hooks.append(debug_hooks.LogPerfHook(args.batch_size))

    if args.elastic:
        from kungfu.tensorflow.experimental.hook import ElasticHook
        elastic_hook = ElasticHook(args.batch_size, args.epochs,
                                   args.epoch_size)
        hooks.append(elastic_hook)

        schedule = parse_scheule(args.resize_schedule)
        profile_resize_hook = debug_hooks.ProfileResizeHook(schedule)
        hooks.append(profile_resize_hook)

        input_fn = build_input_fn(args.batch_size)
        classifier.train(input_fn, hooks=hooks)
    else:
        input_fn = build_input_fn(args.batch_size, args.train_steps)

        sync_step_hook = debug_hooks.SyncStepHook()
        if args.sync_step:
            hooks.append(sync_step_hook)

        _log_event('BEGIN :: classifier.train')
        classifier.train(input_fn, hooks=hooks, max_steps=args.train_steps)
        _log_event('END :: classifier.train')

    _log_event('END :: run_with_estimator')


def main():
    _log_event('BEGIN :: main')
    args = parse_args()
    tf_methods = {
        'simple': run_simple_session,
        'monitored': run_with_session_and_hooks,
        'estimator': run_with_estimator,
    }
    tf_methods[args.tf_method](args)
    _log_event('END :: main')


main()
