import argparse

import tensorflow as tf
from kungfu.python import current_cluster_size, detached
from kungfu.tensorflow.initializer import broadcast_variables
from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
from kungfu.tensorflow.v1.helpers.mnist import load_datasets

MNIST_DATA_SIZE = 60000


def parse_args():
    p = argparse.ArgumentParser(description='Example.')
    p.add_argument('--data-dir', type=str, default='.', help='')
    p.add_argument('--model-dir', type=str, default='.', help='')
    p.add_argument('--kf-optimizer', type=str, default='sync_sgd', help='')
    p.add_argument('--batch-size', type=int, default=100, help='')
    p.add_argument('--num-epochs', type=int, default=1, help='')
    p.add_argument('--learning-rate', type=float, default=0.01, help='')
    return p.parse_args()


def build_ops(args):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    loss = tf.losses.SparseCategoricalCrossentropy()

    opt = tf.keras.optimizers.SGD(args.learning_rate)
    opt = SynchronousSGDOptimizer(opt)
    return model, loss, opt


@tf.function
def sync_model(model, opt):
    broadcast_variables(model.variables)
    broadcast_variables(opt.variables())


@tf.function
def sync_offsets(xs):
    # TODO: use all_reduce with max op
    broadcast_variables(xs)


@tf.function
def training_step(model, loss, opt, images, labels):
    with tf.GradientTape() as tape:
        probs = model(images, training=True)
        loss_value = loss(labels, probs)

    grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return loss_value


@tf.function
def resize_cluster(new_size):
    from kungfu.tensorflow.ops import resize
    return resize(new_size)


def build_dataset(args):
    data = load_datasets(args.data_dir, normalize=True)
    samples = data.train.images.reshape([-1, 28, 28, 1])
    labels = tf.cast(data.train.labels, tf.int32)
    samples = tf.data.Dataset.from_tensor_slices(samples)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((samples, labels))
    ds = ds.batch(args.batch_size)
    ds = ds.shuffle(10000)
    ds = ds.repeat()  # repeat infinitely
    return ds


def train(args):
    step_based_schedule = {
        100: 2,
        200: 3,
        300: 4,
        400: 2,
        500: 3,
        600: 1,
    }
    ds = build_dataset(args)
    model, loss, opt = build_ops(args)
    need_sync = True
    total_samples = int(MNIST_DATA_SIZE * args.num_epochs)
    trained_samples = tf.Variable(0)
    global_step = tf.Variable(0)
    for local_step, (images, labels) in enumerate(ds):
        global_step.assign_add(1)
        trained_samples.assign_add(current_cluster_size() * args.batch_size)
        loss_value = training_step(model, loss, opt, images, labels)
        if need_sync:
            sync_offsets([global_step, trained_samples])
            sync_model(model, opt)
            need_sync = False
        step = int(global_step)
        print('step: %d loss: %f' % (step, loss_value))
        if step in step_based_schedule:
            new_size = step_based_schedule[step]
            need_sync = resize_cluster(new_size)
            if detached():
                break

        if trained_samples >= total_samples:
            break


def check_tf_version():
    major, minor, patch = tf.__version__.split('.')
    assert (major == '2')


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    check_tf_version()
    print('main started')
    main()
    print('main finished')
