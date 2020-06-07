import argparse
import functools
import operator
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.util import deprecation
import tensorflow_datasets as tfds

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


def model_fn(features, labels, mode, config):
    output, predictions = slp(features, 10)
    loss = tf.losses.sparse_softmax_cross_entropy(labels,
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


def input_fn_builder(split, batch_size, epochs=1, shuffle=True):
    def transform(image, label):
        return tf.cast(image, tf.float32) / 255., label

    def input_fn():
        dataset, dataset_info = tfds.load("mnist", as_supervised=True, with_info=True)
        dataset = dataset[split]
        dataset = dataset.map(transform)
        dataset = dataset.repeat(epochs)
        dataset = dataset.cache()
        if shuffle:
            dataset = dataset.shuffle(dataset_info.splits['train'].num_examples)
        dataset = dataset.batch(batch_size)
        return dataset
    
    return input_fn


def get_model_dir(args):
    from kungfu.ext import uid
    x = uid()
    port = (x >> 16) & 0xffff
    version = x & 0xffff
    suffix = '%d.%d' % (port, version)
    return os.path.join(args.model_dir, suffix)


MNIST_DATA_SIZE = 60000


def main(do_eval=True):
    args = parse_args()
    model_dir = get_model_dir(args)

    classifier = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    from kungfu.tensorflow.hook import ElasticHook
    hooks = [ElasticHook(args.batch_size, args.num_epochs, MNIST_DATA_SIZE)]

    classifier.train(input_fn_builder("train",
                              args.batch_size,
                              epochs=args.num_epochs),
                     hooks=hooks)

    if not do_eval:
        import time
        time.sleep(1)
        return
    results = classifier.evaluate(input_fn_builder("test",
                                           args.batch_size,
                                           shuffle=False),
                                  hooks=[],
                                  steps=1)
    print('results: %s' % (results, ))


if __name__ == '__main__':
    print('main started')
    main(False)
    print('main finished')
