import shutil

import kungfu.tensorflow as kf
import tensorflow as tf
from kungfu.tensorflow.v1.helpers import random_input


def build_random_input_fn():
    def input_fn():
        samples, labels = random_input([32, 32, 3], 10)
        samples = tf.data.Dataset.from_tensors(samples)
        labels = tf.data.Dataset.from_tensors(labels)
        features = {'x': samples}
        ds = tf.data.Dataset.zip((features, labels))
        return ds

    return input_fn


def build_model_fn():
    def model_fn(features, labels, mode):
        print('model_fn called')
        x = kf.get_or_create_global_variable('x', [], tf.int64)
        loss = tf.constant(1.0)
        gs = tf.train.get_or_create_global_step()
        train_op = tf.assign_add(gs, x + 1)
        eval_metric_ops = None
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          eval_metric_ops=eval_metric_ops)

    return model_fn


def build_estimator():
    model_dir = 'checkpoints'
    shutil.rmtree(model_dir, ignore_errors=True)
    model_fn = build_model_fn()
    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)
    return classifier


def run_estimator():
    input_fn = build_random_input_fn()
    classifier = build_estimator()
    for i in range(2):
        classifier.train(input_fn=input_fn, max_steps=10)
        print('#%d train finished' % (i))
        classifier.evaluate(input_fn=input_fn, steps=10)
        print('#%d evaluate finished' % (i))


def run_simple_session():
    x = kf.get_or_create_global_variable('xx', shape=[], dtype=tf.int32)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(x)


def test_kungfu_global_variables():
    for i in range(2):
        run_simple_session()
        run_simple_session()


def test_kungfu_global_variables_with_estimator():
    for i in range(2):
        run_estimator()
        run_estimator()
