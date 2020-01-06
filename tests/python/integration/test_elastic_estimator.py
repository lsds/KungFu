import argparse
import os

import tensorflow as tf
from kungfu.tensorflow.ops import (all_reduce, broadcast, counter,
                                   get_init_checkpoint, resize_cluster,
                                   step_based_schedule)
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
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions)
    }
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def input_fn(ds, batch_size, epochs=1):
    features = {'x': ds.images}
    return tf.estimator.inputs.numpy_input_fn(x=features,
                                              y=ds.labels,
                                              batch_size=batch_size,
                                              num_epochs=epochs,
                                              shuffle=False)


class KungFuElasticTrainHook(tf.train.SessionRunHook):
    def __init__(self, schedule, max_step):
        self._schedule = schedule
        self._max_step = max_step
        self._counter = 0
        self._verbose = False
        self.log('SimpleHook created')

    def log(self, msg):
        if self._verbose:
            print(msg)

    def _build_resize_op(self, config, init_step):
        step = counter(init_step)
        new_size = step_based_schedule(config, step)
        ckpt_tensor = tf.as_string(step + 1)
        resize_op = resize_cluster(ckpt_tensor, new_size)
        return resize_op

    def begin(self):
        self.log('begin')
        self._kungfu_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._advance = tf.assign_add(self._kungfu_step, 1)
        vs = tf.global_variables()
        print('%d global variables' % (len(vs)))
        # for v in vs:
        #     print('%s :: %s %s' % (v.name, str(v.dtype), v.shape))
        ckpt = os.getenv('KUNGFU_INIT_CKPT')
        if ckpt is not None:
            self._init_kungfu_step = tf.assign(self._kungfu_step, int(ckpt))
            self._resize_op = self._build_resize_op(self._schedule, int(ckpt))
        else:
            self._init_kungfu_step = None

    def after_create_session(self, sess, coord):
        self.log('after_create_session')
        if self._init_kungfu_step is not None:
            sess.run(self._init_kungfu_step)
        global_step = sess.run(tf.train.get_global_step())
        kungfu_step = sess.run(self._kungfu_step)
        print('session created with global_step: %d, kungfu_step: %d' %
              (global_step, kungfu_step))

    def before_run(self, run_context):
        self.log('before_run: %d' % (self._counter))
        kungfu_step = run_context.session.run(self._kungfu_step)
        if kungfu_step >= self._max_step:
            print('request_stop before kungfu_step: %d' % (kungfu_step))
            # run_context.request_stop()
            # FIXME: force quit

    def after_run(self, run_context, run_values):
        self.log('after_run: %d' % (self._counter))
        self._counter += 1

        kungfu_step = run_context.session.run(self._kungfu_step)

        changed, keep = run_context.session.run(self._resize_op)
        if changed:
            print('changed on %d' % (kungfu_step))
            if not keep:
                run_context.request_stop()

        run_context.session.run(self._advance)
        kungfu_step = run_context.session.run(self._kungfu_step)
        if kungfu_step >= self._max_step:
            print('request_stop on kungfu_step: %d' % (kungfu_step))
            run_context.request_stop()

    def end(self, sess):
        global_step = sess.run(tf.train.get_global_step())
        kungfu_step = sess.run(self._kungfu_step)
        print('stopped at global_step: %d, kungfu_step: %d' %
              (global_step, kungfu_step))
        self.log('end')


def get_model_dir(args):
    self_id = os.getenv('KUNGFU_SELF_SPEC')
    ckpt = os.getenv('KUNGFU_INIT_CKPT')
    uid = '%s@%s' % (self_id, ckpt)  # FIXME: provide an API
    return os.path.join(args.model_dir_prefix, uid)


def main():
    args = parse_args()
    init_step = int(os.getenv('KUNGFU_INIT_CKPT'))
    print('using config: %s, max step=%d, init step: %d' %
          (args.schedule, args.max_step, init_step))

    data = load_datasets(args.data_dir, normalize=True)
    classifier = tf.estimator.Estimator(model_fn,
                                        model_dir=get_model_dir(args))

    classifier.train(
        input_fn(data.train, 1000),
        hooks=[KungFuElasticTrainHook(args.schedule, args.max_step)],
        max_steps=args.max_step)
    results = classifier.evaluate(input_fn(data.test, 1000), hooks=[], steps=1)
    print('results: %s' % (results, ))


main()
