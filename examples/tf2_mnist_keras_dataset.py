import argparse

import tensorflow as tf
from kungfu.python import current_cluster_size, current_rank, run_barrier
from kungfu.tensorflow.optimizers import (PairAveragingOptimizer,
                                          SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer)
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesCallback

parser = argparse.ArgumentParser(description='KungFu mnist example.')
parser.add_argument('--kf-optimizer',
                    type=str,
                    default='sync-sgd',
                    help='available options: sync-sgd, async-sgd, sma')
args = parser.parse_args()

(x_train, y_train), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % current_rank())

train_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(x_train[..., tf.newaxis] / 255.0,
             tf.float32), tf.cast(y_train, tf.int64)))
train_dataset = train_dataset.repeat().shuffle(10000).batch(128)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# KungFu: adjust learning rate based on number of GPUs.
opt = tf.keras.optimizers.SGD(0.001 * current_cluster_size())
# opt = tf.compat.v1.train.AdamOptimizer(0.001 * current_cluster_size())

if args.kf_optimizer == 'sync-sgd':
    opt = SynchronousSGDOptimizer(opt)
elif args.kf_optimizer == 'async-sgd':
    opt = PairAveragingOptimizer(opt)
elif args.kf_optimizer == 'sma':
    opt = SynchronousAveragingOptimizer(opt)
else:
    raise RuntimeError('Unknown KungFu optimizer')

mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                    optimizer=opt,
                    metrics=['accuracy'])

# KungFu: insert the global variable broadcast callback.
callbacks = [
    BroadcastGlobalVariablesCallback(),
]

# KungFu: write logs on worker 0.
verbose = 1 if current_rank() == 0 else 0

# Train the model.
# KungFu: adjust number of steps based on number of GPUs.
mnist_model.fit(train_dataset,
                steps_per_epoch=500 // current_cluster_size(),
                epochs=1,
                callbacks=callbacks,
                verbose=verbose)

# KungFu: run evaluation after all finishes training.
run_barrier()

_, (x_test, y_test) = \
    tf.keras.datasets.mnist.load_data()
test_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(x_test[..., tf.newaxis] / 255.0,
             tf.float32), tf.cast(y_test, tf.int64)))
test_dataset = test_dataset.batch(128)

print('\n# Evaluate on test data')
results = mnist_model.evaluate(test_dataset)
print('test loss, test acc:', results)
