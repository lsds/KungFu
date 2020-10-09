import argparse
import functools
import operator
import os

import numpy as np
import tensorflow as tf
from kungfu.tensorflow.policy import PolicyHook
from kungfu.tensorflow.v1.helpers.mnist import load_datasets
from tensorflow.python.util import deprecation

from elastic_policy import ScheduledElasticPolicy

deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_args():
    p = argparse.ArgumentParser(description='Example.')
    p.add_argument('--data-dir', type=str, default='.', help='')
    p.add_argument('--model-dir', type=str, default='.', help='')
    p.add_argument('--kf-optimizer', type=str, default='sync_sgd', help='')
    p.add_argument('--batch-size', type=int, default=100, help='')
    p.add_argument('--num-epochs', type=int, default=1, help='')
    p.add_argument('--learning-rate', type=float, default=0.01, help='')
    return p.parse_args()


def slp(x, logits):
    n = functools.reduce(operator.mul, [int(d) for d in x.shape[1:]], 1)
    output = tf.layers.dense(inputs=tf.reshape(x, [-1, n]), units=logits)
    return output, tf.argmax(output, axis=1)


def model_fn(features, labels, mode):
    output, predictions = slp(features['x'], 10)
    loss = tf.losses.sparse_softmax_cross_entropy(tf.cast(labels, tf.int32),
                                                  output)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions)
    }
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
    optimizer = SynchronousSGDOptimizer(optimizer)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def input_fn(ds, batch_size, epochs=1, shuffle=True):
    features = {'x': ds.images}
    return tf.estimator.inputs.numpy_input_fn(x=features,
                                              y=ds.labels,
                                              batch_size=batch_size,
                                              num_epochs=epochs,
                                              shuffle=shuffle)


def get_model_dir(args):
    from kungfu.python import uid
    x = uid()
    port = (x >> 16) & 0xffff
    version = x & 0xffff
    suffix = '%d.%d' % (port, version)
    return os.path.join(args.model_dir, suffix)


MNIST_DATA_SIZE = 60000


def main(do_eval=True):
    args = parse_args()
    model_dir = get_model_dir(args)

    data = load_datasets(args.data_dir, normalize=True)
    classifier = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    # resize cluster at given step
    step_schedule = {
        10: 2,
        20: 3,
        30: 4,
        40: 1,
        50: 1,
    }

    policy = ScheduledElasticPolicy(step_schedule)
    hooks = [
        PolicyHook([policy], MNIST_DATA_SIZE, args.num_epochs,
                   args.batch_size),
    ]

    classifier.train(input_fn(data.train,
                              args.batch_size,
                              epochs=args.num_epochs),
                     hooks=hooks)

    if not do_eval:
        import time
        time.sleep(1)
        return
    results = classifier.evaluate(input_fn(data.test,
                                           args.batch_size,
                                           shuffle=False),
                                  hooks=[],
                                  steps=1)
    print('results: %s' % (results, ))


if __name__ == '__main__':
    print('main started')
    main(False)
    print('main finished')
