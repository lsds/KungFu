import tensorflow as tf
from kungfu import current_cluster_size, current_rank
from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
from kungfu.tensorflow.v2.initializer import BroadcastGlobalVariablesCallback

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % current_rank())

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0,
             tf.float32), tf.cast(mnist_labels, tf.int64)))
dataset = dataset.repeat().shuffle(10000).batch(128)

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
opt = tf.compat.v1.train.AdamOptimizer(0.001 * current_cluster_size())

# KungFu: wrap tf.compat.v1.train.Optimizer.
opt = SynchronousSGDOptimizer(opt)

mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                    optimizer=opt,
                    metrics=['accuracy'])

callbacks = [
    BroadcastGlobalVariablesCallback(),
]

# KungFu: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if current_rank() == 0:
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

# KungFu: write logs on worker 0.
verbose = 1 if current_rank() == 0 else 0

# Train the model.
# KungFu: adjust number of steps based on number of GPUs.
mnist_model.fit(dataset,
                steps_per_epoch=500 // current_cluster_size(),
                callbacks=callbacks,
                epochs=24,
                verbose=verbose)
