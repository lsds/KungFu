import argparse
import os

import numpy as np
import tensorflow as tf
from kungfu.tensorflow.v1.helpers.mnist import load_datasets


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
    self_id = os.getenv('KUNGFU_SELF_SPEC')
    ckpt = os.getenv('KUNGFU_INIT_STEP')
    uid = '%s@%s' % (self_id, ckpt)  # FIXME: provide an API
    return os.path.join(args.model_dir, uid)


MNIST_DATA_SIZE = 60000


def main():
    args = parse_args()
    # print('using config: %s, max step=%d' % (args.schedule, args.max_step))
    model_dir = get_model_dir(args)

    data = load_datasets(args.data_dir, normalize=True)
    classifier = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    batch_size = 100
    epochs = 10
    max_steps = MNIST_DATA_SIZE * epochs / batch_size
    print('max_steps: %d' % (max_steps))

    from kungfu.tensorflow.experimental.hook import ElasticHook
    hooks = [ElasticHook(max_steps)]

    classifier.train(input_fn(data.train, batch_size, epochs=epochs),
                     hooks=hooks,
                     max_steps=max_steps)
    print('train finished')

    results = classifier.evaluate(input_fn(data.test,
                                           batch_size,
                                           shuffle=False),
                                  hooks=[],
                                  steps=1)
    print('results: %s' % (results, ))


if __name__ == '__main__':
    main()
