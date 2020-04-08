from __future__ import absolute_import, division, print_function
from tensorflow.python.util import deprecation
from kungfu import (current_rank, current_cluster_size)

import gzip
import os
import shutil
import tempfile
import importlib.util

import numpy as np
import tensorflow as tf
from six.moves import urllib

home = os.getenv("HOME")

flags = tf.app.flags
flags.DEFINE_string(
    'data_dir', './mnist/data', 'Directory where mnist data will be downloaded'
    ' if the data is not already there')
flags.DEFINE_string('model_dir', './mnist/model',
                    'Directory where all models are saved')
flags.DEFINE_string('log_dir', '/tmp/tensorflow/mnist_estimator/logs/mnist_estimator_with_summaries',
                    'Directory where all logs are saved')
flags.DEFINE_string('kf_optimizer', 'sync_sgd', 'KungFu optimizer')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('num_epochs', 5, 'Num of batches to train (epochs).')
flags.DEFINE_float('learning_rate', 0.001, 'Learning Rate')
flags.DEFINE_bool('ada', False, 'Synchronization primitive adaptation')

FLAGS = flags.FLAGS

dataset_size = 60000

deprecation._PRINT_DEPRECATION_WARNINGS = False

def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_images, unused
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' %
                             (magic, f.name))
        if rows != 28 or cols != 28:
            raise ValueError(
                'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                (f.name, rows, cols))


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_items, unused
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' %
                             (magic, f.name))


def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, \
        tf.gfile.Open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


def dataset(directory, images_file, labels_file):
    """Download and parse MNIST dataset."""

    images_file = download(directory, images_file)
    labels_file = download(directory, labels_file)

    check_image_file_header(images_file)
    check_labels_file_header(labels_file)

    def decode_image(image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image / 255.0

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.to_int32(label)

    images = tf.data.FixedLengthRecordDataset(
        images_file, 28 * 28, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(labels_file, 1,
                                              header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


def train(directory):
    """tf.data.Dataset object for MNIST training data."""
    return dataset(directory, 'train-images-idx3-ubyte',
                   'train-labels-idx1-ubyte')


def test(directory):
    """tf.data.Dataset object for MNIST test data."""
    return dataset(directory, 't10k-images-idx3-ubyte',
                   't10k-labels-idx1-ubyte')


def train_data():
    data = train(FLAGS.data_dir)
    data = data.batch(FLAGS.batch_size)
    data = data.repeat()
    
    # shuffle data amongst workers 
    data = data.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    cluster_size = current_cluster_size()
    idx = current_rank()
    data = data.shard(num_shards=cluster_size, index=idx)
    return data


def eval_data():
    data = test(FLAGS.data_dir)
    data = data.cache()
    data = data.batch(FLAGS.batch_size)
    return data


def lenet():
    layers = tf.keras.layers

    model = tf.keras.Sequential([
        layers.Reshape(target_shape=[28, 28, 1], input_shape=(28 * 28, )),
        layers.Conv2D(filters=20,
                      kernel_size=[5, 5],
                      padding='same',
                      activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=[2, 2]),
        layers.Conv2D(filters=50,
                      kernel_size=[5, 5],
                      padding='same',
                      activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2]),
        layers.Flatten(),
        layers.Dense(units=500, activation=tf.nn.relu),
        layers.Dense(units=10),
    ])

    return model


def model_function(features, labels, mode):
    # get the model
    model = lenet()

    if mode == tf.estimator.ModeKeys.TRAIN:
        # pass the input through the model
        logits = model(features)

        # get the cross-entropy loss and name it
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits)
        tf.identity(loss, 'train_loss')
        tf.summary.scalar('TrainLoss', loss)

        # record the accuracy and name it
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=tf.argmax(logits, axis=1))
        tf.identity(accuracy[1], name='train_accuracy')

        tf.summary.scalar('TrainAccuracy', accuracy[1])

        # use Adam to optimize
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        tf.identity(FLAGS.learning_rate, name='learning_rate')

        #get custom Ada optimizer
        #TODO: fix this and enable it
        

        # Create a hook to print acc, loss & global step every 100 iter.   
        # train_hook_list= []
        # train_tensors_log = {'TrainAccuracy': accuracy[1],
        #                     'TrainLoss': loss}
        # train_hook_list.append(tf.train.LoggingTensorHook(
        #     tensors=train_tensors_log, every_n_iter=1))

        #KungFu: Wrap the tf.train.optimizer with KungFu optimizers
        if FLAGS.ada:
            spec = importlib.util.spec_from_file_location("CustomAdaSGD", 
            os.path.join(home,"KungFu/experimental/net_monitoring/CustomAdaSGD.py")) 
            CustomAda = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(CustomAda)

            optimizer = CustomAda.CustomAdaSGDOptimizer(optimizer)
        else:
            if FLAGS.kf_optimizer == 'sync_sgd':
                from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
                optimizer = SynchronousSGDOptimizer(optimizer)
            elif FLAGS.kf_optimizer == 'async_sgd':
                from kungfu.tensorflow.optimizers import PairAveragingOptimizer
                optimizer = PairAveragingOptimizer(optimizer)
            else:
                raise RuntimeError('Unknown kungfu optimizer')

        # create an estimator spec to optimize the loss
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss,
                                        tf.train.get_or_create_global_step()))

    elif mode == tf.estimator.ModeKeys.EVAL:
        # pass the input through the model
        logits = model(features, training=False)

        # get the cross-entropy loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits)

        # use the accuracy as a metric
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=tf.argmax(logits, axis=1))

        # create an estimator spec with the loss and accuracy
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={'EvalAccuracy': accuracy})

    return estimator_spec

def main(_):
    # print(FLAGS.model_dir)
    # print(FLAGS.data_dir)

    # if current_rank() == 0:
    save_checkpoints_steps = 100
    save_summary_steps = 1
    model_dir = FLAGS.model_dir + "/"+str(current_rank())
    config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        save_summary_steps=save_summary_steps)
    # else: 
    #     config=None

    mnist_classifier = tf.estimator.Estimator(model_fn=model_function,
                                            model_dir=model_dir,
                                            config=config)

    # TODO: CAUTION: number calculated based on batch size and is needed to prevent any fault
    # caused by dataset sharding 
    # num_train_steps = 1000
    num_train_steps = (FLAGS.num_epochs * dataset_size)/(current_cluster_size() * FLAGS.batch_size)
    
    #from ./experimental.net_monitoring.NetMonHook import NetMonHook
    if FLAGS.ada:
        spec = importlib.util.spec_from_file_location("NetMonHook", os.path.join(home,"repos/KungFu/experimental/net_monitoring/NetMonHook.py"))
        NetMon = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(NetMon)

        hooks=[NetMon.NetMonHook(FLAGS.log_dir)]
    else:
        hooks = None

    # from tensorflow.python.training.basic_session_run_hooks import StepCounterHook
    # hooks=[StepCounterHook(8)]

    train_spec = tf.estimator.TrainSpec(input_fn=train_data, max_steps=num_train_steps, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_data)
    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()