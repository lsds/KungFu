import tensorflow as tf
from kungfu import current_cluster_size, current_rank, run_barrier
from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer, PairAveragingOptimizer, SynchronousAveragingOptimizer
from kungfu.tensorflow.v2.initializer import BroadcastGlobalVariablesCallback

flags = tf.compat.v1.flags
flags.DEFINE_string('kf-optimizer', 'sync-sgd', 'KungFu optimizer')
FLAGS = flags.FLAGS

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# KungFu: sharding your data using local rank.
epochs = 10
train_dataset = train_dataset.shard(
    current_cluster_size(),
    current_rank()).repeat(epochs).shuffle(10000).batch(128)
test_dataset = test_dataset.batch(128)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# KungFu: adjust learning rate based on number of GPUs.
opt = tf.compat.v1.train.AdamOptimizer(0.001 * current_cluster_size())

# KungFu: wrap tf.compat.v1.train.Optimizer.
if FLAGS.kf_optimizer == 'sync-sgd':
    opt = SynchronousSGDOptimizer(opt)
elif FLAGS.kf_optimizer == 'async-sgd':
    opt = PairAveragingOptimizer(opt)
elif FLAGS.kf_optimizer == 'sma':
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
mnist_model.fit(train_dataset, callbacks=callbacks, verbose=verbose)

# KungFu: run evaluation after all finishes training.
run_barrier()
print('\n# Evaluate on test data')
results = mnist_model.evaluate(test_dataset)
print('test loss, test acc:', results)
