import argparse
import os

import tensorflow as tf
from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
from kungfu.tensorflow.v1.helpers.mnist import load_datasets
from tensorflow.python.util import deprecation
from kungfu.tensorflow.hooks import KungFuElasticTrainHook

deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_args():
    p = argparse.ArgumentParser(description='Example.')
    p.add_argument('--schedule',
                   type=str,
                   default='3:2,3:4,3:16,3:1',
                   help='cluster size schedule')
    p.add_argument('--max-step', type=int, default=10, help='max train step')
    p.add_argument('--data-dir', type=str, default='.')
    p.add_argument('--model-dir-prefix', type=str, default='./checkpoints/')
    return p.parse_args()


def slp(x, logits):
    n = 1
    for d in x.shape[1:]:
        n *= int(d)
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
    return os.path.join(args.model_dir_prefix, suffix)


def main():
    args = parse_args()
    print('using config: %s, max step=%d' % (args.schedule, args.max_step))
    model_dir = get_model_dir(args)

    data = load_datasets(args.data_dir, normalize=True)
    classifier = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    classifier.train(input_fn(data.train, 1000),
                     hooks=[
                         KungFuElasticTrainHook(args.schedule, args.max_step,
                                                model_dir)
                     ],
                     max_steps=args.max_step)

    results = classifier.evaluate(input_fn(data.test, 1000, shuffle=False),
                                  hooks=[],
                                  steps=1)
    print('results: %s' % (results, ))


main()
