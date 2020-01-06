import argparse
import os

import numpy as np
import tensorflow as tf
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import (all_reduce, broadcast, consensus, counter,
                                   get_init_checkpoint, resize_cluster,
                                   step_based_schedule)
from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
from kungfu.tensorflow.v1.helpers.mnist import load_datasets
from tensorflow.python.util import deprecation

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


class KungFuElasticTrainHook(tf.train.SessionRunHook):
    def __init__(self, schedule, max_step, model_dir):
        self._schedule = schedule
        self._max_step = max_step
        self._model_dir = model_dir
        self._need_sync = True

    def _build_resize_op(self, config, init_step):
        step = counter(init_step)
        new_size = step_based_schedule(config, step)
        ckpt_tensor = tf.as_string(step + 1)
        resize_op = resize_cluster(ckpt_tensor, new_size)
        return resize_op

    def begin(self):
        self._kungfu_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._advance = tf.assign_add(self._kungfu_step, 1)
        self._sync_op = BroadcastGlobalVariablesOp()
        ckpt = os.getenv('KUNGFU_INIT_CKPT')
        self._init_kungfu_step = tf.assign(self._kungfu_step, int(ckpt))
        self._resize_op = self._build_resize_op(self._schedule, int(ckpt))
        self._reset_global_step = tf.assign(tf.train.get_global_step(),
                                            int(ckpt))

    def after_create_session(self, sess, coord):
        sess.run(self._init_kungfu_step)
        sess.run(self._reset_global_step)

    def before_run(self, run_context):
        kungfu_step = run_context.session.run(self._kungfu_step)
        if kungfu_step >= self._max_step:
            print('request_stop before kungfu_step: %d' % (kungfu_step))
            # run_context.request_stop()
            # FIXME: force quit

        if self._need_sync:
            run_context.session.run(self._sync_op)
            self._need_sync = False

    def after_run(self, run_context, run_values):
        kungfu_step = run_context.session.run(self._kungfu_step)
        changed, keep = run_context.session.run(self._resize_op)
        if changed:
            print('changed on %d' % (kungfu_step))
            self._need_sync = True
            if not keep:
                run_context.request_stop()
                return

        kungfu_step = run_context.session.run(self._advance)
        if kungfu_step >= self._max_step:
            print('request_stop on kungfu_step: %d' % (kungfu_step))
            run_context.request_stop()

    def end(self, sess):
        global_step = sess.run(tf.train.get_global_step())
        kungfu_step = sess.run(self._kungfu_step)
        print('stopped at global_step: %d, kungfu_step: %d' %
              (global_step, kungfu_step))

        self.save(sess, 'final')

    def save(self, sess, idx):
        vs = tf.global_variables()
        d = dict()
        for t in vs:
            v = sess.run(t)
            d[t.name] = v
        np.savez(os.path.join(self._model_dir, 'variables-%s.npz' % (idx)),
                 **d)


def get_model_dir(args):
    self_id = os.getenv('KUNGFU_SELF_SPEC')
    ckpt = os.getenv('KUNGFU_INIT_CKPT')
    uid = '%s@%s' % (self_id, ckpt)  # FIXME: provide an API
    return os.path.join(args.model_dir_prefix, uid)


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
