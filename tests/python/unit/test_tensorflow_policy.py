import shutil

import kungfu.tensorflow as kf
import tensorflow as tf
from kungfu.tensorflow.v1.helpers import random_input
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

from kungfu.tensorflow.policy import PolicyHook

from scheduled_batch_size_policy import ScheduledBatchSizePolicyExample


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
        loss = tf.constant(1.0)
        gs = tf.train.get_or_create_global_step()
        train_op = tf.assign_add(gs, 1)
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


def run_policy_example_1():
    input_fn = build_random_input_fn()
    classifier = build_estimator()
    schedule = {
        10: 32,
        20: 64,
        30: 128,
    }
    policy = ScheduledBatchSizePolicyExample(schedule)
    policy_hook = PolicyHook([policy],
                             epoch_size=100,
                             epoch_num=10,
                             init_batch_size=16)
    classifier.train(input_fn=input_fn, max_steps=100, hooks=[policy_hook])


def test_1():
    run_policy_example_1()


def main():
    run_policy_example_1()


if __name__ == "__main__":
    main()
