#
# This file is a simplified version of the Horovod Keras example:
# https://github.com/uber/horovod/blob/master/examples/keras_imagenet_resnet50.py
#
from __future__ import print_function

import argparse
import keras
from keras import backend as K
from keras.preprocessing import image
import tensorflow as tf
import kungfu as kf
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
parser.add_argument('--verbose', type=bool, default=False, help='print log')
parser.add_argument('--cluster-size', type=int, default=1, help='cluster size')
parser.add_argument('--local-rank', type=int, default=1, help='local rank')

args = parser.parse_args()

# TODO: make the cluster size discovered by the peer.
kf_size = args.cluster_size
kf_local_rank = args.local_rank

# Kungfu: pin GPU to be used to process local rank (one GPU per process)
# TODO: here is a static binding of learners and GPUs. However, in a fully elastic cluster,
# shall we use a dynamic learning task dispatching architecture?
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(kf_local_rank)
K.set_session(tf.Session(config=config))

# Training data iterator.
train_gen = image.ImageDataGenerator(
    width_shift_range=0.33, height_shift_range=0.33, zoom_range=0.5, horizontal_flip=True,
    preprocessing_function=keras.applications.resnet50.preprocess_input)
train_iter = train_gen.flow_from_directory(args.train_dir,
                                           batch_size=args.batch_size,
                                           target_size=(224, 224))

# Set up standard ResNet-50 model.
model = keras.applications.resnet50.ResNet50(weights=None)

# ResNet-50 model that is included with Keras is optimized for inference.
# Add L2 weight decay & adjust BN settings.
model_config = model.get_config()
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

# FIXME: kf_size needs to be logical data parallelism
opt = keras.optimizers.SGD(lr=args.base_lr * kf_size, momentum=args.momentum)

# Kungfu: add Distributed Optimizer.
opt = kf.AsyncSGDOptimizer(opt)

model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy', 'top_k_categorical_accuracy'])

# Train the model. The training will randomly sample 1 / N batches of training data.
# TODO: ensure that all worker starts from the same initial weighst
model.fit_generator(train_iter,
                    steps_per_epoch=len(train_iter) // kf_size, # FIXME: kf_size needs to be logical data parallelism
                    epochs=args.epochs,
                    verbose=args.verbose,
                    workers=4)
