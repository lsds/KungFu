#
# ResNet-50 model training using Keras and Horovod.
#
# This model is an example of a computation-intensive model that achieves good accuracy on an image
# classification task.  It brings together distributed training concepts such as learning rate
# schedule adjustments with a warmup, randomized data reading, and checkpointing on the first worker
# only.
#
# Note: This model uses Keras native ImageDataGenerator and not the sophisticated preprocessing
# pipeline that is typically used to train state-of-the-art ResNet-50 model.  This results in ~0.5%
# increase in the top-1 validation error compared to the single-crop top-1 validation error from
# https://github.com/KaimingHe/deep-residual-networks.
#
from __future__ import print_function

import argparse
import keras
from keras import backend as K
from keras.preprocessing import image
import tensorflow as tf
import horovod.keras as hvd
import os

parser = argparse.ArgumentParser(description='Keras ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.h5',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

args = parser.parse_args()

# Parsing protocol buffer-saved training data as in the following blog post
# https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36
def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                        "label": tf.FixedLenFeature([], tf.int64)}
    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)
    # Turn your saved image string into an array
    parsed_features['image'] = tf.decode_raw(
        parsed_features['image'], tf.float32)
    return parsed_features['image'], parsed_features["label"]

def create_dataset(filepath, batch_size):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    # This dataset will go on forever
    dataset = dataset.repeat()
    # Set the number of datapoints you want to load and shuffle
    #dataset = dataset.shuffle(SHUFFLE_BUFFER)
    # Set the batchsize
    dataset = dataset.batch(batch_size)
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    # Create your tf representation of the iterator
    image, label = iterator.get_next()
    # Bring your picture back in shape
    image = tf.reshape(image, [-1, 256, 256, 1])
    # Create a one hot array for your labels
    label = tf.one_hot(label, batch_size)
    label = tf.reshape(label, [1, batch_size])

    return image, label

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

# Reading data from blob file 'test-data'
image, label = create_dataset('test-data', args.batch_size)

model_input = keras.layers.Input(tensor=image)

# Set up standard ResNet-50 model.
model = keras.applications.resnet50.ResNet50(weights=None, input_tensor=model_input)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast both model and optimizer weights
# to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    model = hvd.load_model(args.checkpoint_format.format(epoch=resume_from_epoch),
                           compression=compression)
else:
    # ResNet-50 model that is included with Keras is optimized for inference.
    # Add L2 weight decay & adjust BN settings.
    model_config = model.get_config()
    #import pdb; pdb.set_trace()
    for layer, layer_config in zip(model.layers, model_config['layers']):
        if hasattr(layer, 'kernel_regularizer'):
            regularizer = keras.regularizers.l2(args.wd)
            layer_config['config']['kernel_regularizer'] = \
                {'class_name': regularizer.__class__.__name__,
                 'config': regularizer.get_config()}
        if type(layer) == keras.layers.BatchNormalization:
            layer_config['config']['momentum'] = 0.9
            layer_config['config']['epsilon'] = 1e-5

    model = keras.models.Model.from_config(model_config)

    # Horovod: adjust learning rate based on number of GPUs.
    opt = keras.optimizers.SGD(lr=args.base_lr * hvd.size(),
                               momentum=args.momentum)

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt, compression=compression)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose),

    # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, end_epoch=30, multiplier=1.),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=30, end_epoch=60, multiplier=1e-1),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=60, end_epoch=80, multiplier=1e-2),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=80, multiplier=1e-3),
]

# Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format))
    callbacks.append(keras.callbacks.TensorBoard(args.log_dir))

# Train the model. The training will randomly sample 1 / N batches of training data and
# 3 / N batches of validation data on every worker, where N is the number of workers.
# Over-sampling of validation data helps to increase probability that every validation
# example will be evaluated.
model.fit(y=label,
          # steps_per_epoch=len(image) // hvd.size(),
          callbacks=callbacks,
          epochs=args.epochs,
          verbose=verbose)
