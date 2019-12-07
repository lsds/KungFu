from __future__ import absolute_import, division, print_function

import errno
import os

import numpy as np
import tensorflow as tf
from kungfu import current_cluster_size, current_rank
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesHook
from kungfu.tensorflow.ops import resize_cluster
from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
from tensorflow import keras

tf.logging.set_verbosity(tf.logging.INFO)


class ElasticTrainingHook(tf.train.SessionRunHook):
    def __init__(self, epoch_schedule, epoch_size, device_batch_size):
        self._last_step, self._resize_step_schedule = self._parse_epoch_schedule(
            epoch_schedule, epoch_size, device_batch_size)
        print(self._last_step)
        print(self._resize_step_schedule)

    def _parse_epoch_schedule(self, epoch_schedule, epoch_size,
                              device_batch_size):
        """Translate the epoch schedule into a step schedule

        Example:
        epoch_size = 20
        device_batch_size = 1
        epoch_schedule = [(0, 2), (2, 4), (4, 2), (6, 0)]

        resize_step_schedule = [(20, 4), (30, 2)]
        last_step = 50
        """
        resize_step = 0
        resize_step_schedule = []
        for i in range(1, len(epoch_schedule)):
            prev_epoch = epoch_schedule[i - 1][0]
            prev_num_devices = epoch_schedule[i - 1][1]
            prev_epoch_step_count = (epoch_schedule[i][0] -
                                     prev_epoch) * epoch_size // (
                                         prev_num_devices * device_batch_size)
            resize_step += prev_epoch_step_count
            num_devices = epoch_schedule[i][1]
            resize_step_schedule.append((resize_step, num_devices))

        return resize_step, resize_step_schedule[:-1]

        def begin(self):
            self._global_step_tensor = tf.train.get_or_create_global_step()
            if self._global_step_tensor is None:
                raise RuntimeError(
                    "Global step should be created to use ElasticTrainingHook."
                )

        def before_run(self, run_context):
            return tf.estimator.SessionRunArgs(self._global_step_tensor)

        def after_run(self, run_context, run_values):
            global_step = run_values.results + 1
            if global_step >= self._last_step:
                print("ElasticTrainingHook: global step = %s" %
                      str(global_step))
                run_context.request_stop()


flags = tf.app.flags
flags.DEFINE_integer('batch_size', 512, 'Batch size.')
flags.DEFINE_float('num_epochs', 1.0, 'Num of batches to train (epochs).')
FLAGS = flags.FLAGS


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat,
                            units=1024,
                            activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                           logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001,
                                               momentum=0.9)

        # KungFu: add Distributed Optimizer.
        optimizer = SynchronousSGDOptimizer(optimizer)

        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy":
        tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Keras automatically creates a cache directory in ~/.keras/datasets for
    # storing the downloaded MNIST data. This creates a race
    # condition among the workers that share the same filesystem. If the
    # directory already exists by the time this worker gets around to creating
    # it, ignore the resulting exception and continue.
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
    if not os.path.exists(cache_dir):
        try:
            os.mkdir(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
                pass
            else:
                raise

    # Download and load MNIST dataset.
    (train_data, train_labels), (eval_data, eval_labels) = \
        keras.datasets.mnist.load_data()

    # The shape of downloaded data is (-1, 28, 28), hence we need to reshape it
    # into (-1, 784) to feed into our network. Also, need to normalize the
    # features between 0 and 1.
    train_data = np.reshape(train_data, (-1, 784)) / 255.0
    eval_data = np.reshape(eval_data, (-1, 784)) / 255.0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # KungFu: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    model_dir = './mnist_convnet_model' if current_rank() == 0 else None

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_dir,
        config=tf.estimator.RunConfig(session_config=config))

    # KungFu: BroadcastGlobalVariablesHook broadcasts initial variable states from
    # rank 0 to all other processes. This is necessary to ensure consistent
    # initialization of all workers when training is started with random weights or
    # restored from a checkpoint.
    bcast_hook = BroadcastGlobalVariablesHook()
    elastic_hook = ElasticTrainingHook(
        FLAGS.num_epochs, FLAGS.batch_size * current_cluster_size())

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=FLAGS.batch_size,
        num_epochs=None,
        shuffle=True)

    # KungFu: adjust number of steps based on number of GPUs.
    mnist_classifier.train(input_fn=train_input_fn,
                           hooks=[bcast_hook, elastic_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                       y=eval_labels,
                                                       num_epochs=1,
                                                       shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
