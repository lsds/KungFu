import argparse

import tensorflow as tf
from tensorflow.python.util import deprecation

import debug_hooks

deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_args():
    p = argparse.ArgumentParser(
        description='TensorFlow Synthetic Benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model',
                   type=str,
                   default='ResNet50',
                   help='model to benchmark')
    p.add_argument('--batch-size', type=int, default=32, help='batch size')
    p.add_argument('--train-steps',
                   type=int,
                   default=1000,
                   help='number of batches per benchmark iteration')
    return p.parse_args()


def build_optimizer(learning_rate=0.01):
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
    opt = SynchronousSGDOptimizer(opt)
    return opt


def build_input_fn(batch_size, steps=None):
    def input_fn():
        data = tf.random_uniform([batch_size, 224, 224, 3])
        label = tf.random_uniform([batch_size, 1],
                                  minval=0,
                                  maxval=999,
                                  dtype=tf.int64)

        data = tf.data.Dataset.from_tensors(data)
        label = tf.data.Dataset.from_tensors(label)
        ds = tf.data.Dataset.zip((data, label))
        ds = ds.repeat(steps)
        return ds

    return input_fn


def build_train_op(model, features, labels):

    from tensorflow.keras import applications
    model = getattr(applications, model)(weights=None)

    logits = model(features['x'], training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    opt = build_optimizer()
    train_op = opt.minimize(loss)

    return train_op


def build_estimator(model):
    def model_fn(features, labels, mode):
        # output, predictions = slp(features['x'], 10)
        # loss = tf.losses.sparse_softmax_cross_entropy(tf.cast(labels, tf.int32),
        #                                               output)
        # optimizer = tf.train.GradientDescentOptimizer(0.1)
        # train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        # eval_metric_ops = {
        #     'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions)
        # }
        # return tf.estimator.EstimatorSpec(mode=mode,
        #                                   loss=loss,
        #                                   train_op=train_op,
        #                                   eval_metric_ops=eval_metric_ops)

        from tensorflow.keras import applications
        model = getattr(applications, model)(weights=None)

        logits = model(features['x'], training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

        opt = build_optimizer()
        train_op = opt.minimize(loss)

        eval_metric_ops = None
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          eval_metric_ops=eval_metric_ops)

    return model_fn


def run_simple_session(args):
    input_fn = build_input_fn(args.batch_size, args.train_steps)
    ds = input_fn()
    it = ds.make_initializable_iterator()
    get_next = it.get_next()

    samples, labels = get_next
    print(samples)
    print(labels)

    features = {'x': samples}
    train_op = build_train_op(args.model, features, labels)

    with tf.Session() as sess:
        sess.run(it.initializer)
        for _ in range(args.train_steps):
            sess.run(train_op)
            # x, y = sess.run(get_next)
            # print(x.shape)
            # print(y.shape)


def run_with_session_and_hooks(args):
    input_fn = build_input_fn(args.batch_size, args.train_steps)
    ds = input_fn()
    it = ds.make_initializable_iterator()
    get_next = it.get_next()

    samples, labels = get_next
    print(samples)
    print(labels)

    features = {'x': samples}
    train_op = build_train_op(args.model, features, labels)

    # init = tf.global_variables_initializer()

    # with tf.Session() as sess:
    #     # sess.run(init)
    #     sess.run(it.initializer)
    #     for _ in range(10):
    #         x, y = sess.run(get_next)
    #         print(x.shape)
    #         print(y.shape)

    hooks = [
        debug_hooks.LogStepHook(),
    ]

    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        sess.run(it.initializer)
        while sess.should_stop:
            # sess.run(train_op)
            sess.run(train_op)
            x, y = sess.run(get_next)
            print(x.shape)
            print(y.shape)


def run_with_estimator():
    # TODO
    pass


def main():
    args = parse_args()
    # mf = build_model_fn(args.model)
    # run_with_session_and_hooks(args)
    run_simple_session(args)


main()
