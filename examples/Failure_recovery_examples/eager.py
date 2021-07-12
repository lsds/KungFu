#kungfu-run -np 2 -auto-recover 10s python3 examples/Failure_recovery_examples/eager.py --num-epochs 5 --data-dir ./mnist --model-dir checkpoints --batch-size 32 --monitor
import argparse
import tensorflow as tf
import os
from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.initializer import broadcast_variables
from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
from kungfu.tensorflow.v1.helpers.mnist import load_datasets
from kungfu.cmd import monitor_batch_begin, monitor_batch_end, monitor_train_end, monitor_epoch_end
MNIST_DATA_SIZE = 60000


def parse_args():
    p = argparse.ArgumentParser(description='Example.')
    p.add_argument('--data-dir', type=str, default='.', help='')
    p.add_argument('--model-dir', type=str, default='.', help='')
    p.add_argument('--kf-optimizer', type=str, default='sync_sgd', help='')
    p.add_argument('--batch-size', type=int, default=100, help='')
    p.add_argument('--num-epochs', type=int, default=1, help='')
    p.add_argument('--learning-rate', type=float, default=0.01, help='')
    p.add_argument('--monitor', action='store_true', default=False, help='')
    p.add_argument('--restart', type=int, default=0, help='')
    p.add_argument('--save-epoch', type=int, default=1, help='')
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
def training_step(model, loss, opt, images, labels):
    with tf.GradientTape() as tape:
        probs = model(images, training=True)
        loss_value = loss(labels, probs)
    grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value


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
    shard_id = current_rank()
    ds = build_dataset(args)
    model, loss, opt = build_ops(args)
    total_samples = int(MNIST_DATA_SIZE * args.num_epochs)
    trained_samples = tf.Variable(0)
    epochs = 0
    global_step = tf.Variable(0)
    #get checkpoint save path
    file_name = os.path.basename(__file__)
    stem, suffix = os.path.splitext(file_name)
    save_dir_be = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.exists(save_dir_be) and shard_id == 0:
        os.mkdir(save_dir_be)
    save_dir = os.path.join(save_dir_be, stem)
    if not os.path.exists(save_dir) and shard_id == 0:
        os.mkdir(save_dir)
    filepath = "model_" + str(shard_id) + ".h5"
    savepath = os.path.join(save_dir, filepath)
    if (os.path.exists(savepath) and args.restart == 1):
        model = tf.keras.models.load_model(savepath)
    for local_step, (images, labels) in enumerate(ds):
        if args.monitor:
            monitor_batch_begin()
        global_step.assign_add(1)
        trained_samples.assign_add(current_cluster_size() * args.batch_size)
        loss_value = training_step(model, loss, opt, images, labels)
        step = int(global_step)
        print('step: %d loss: %f' % (step, loss_value))
        if args.monitor:
            if trained_samples >= MNIST_DATA_SIZE * (epochs + 1):
                epochs = epochs + 1
                if epochs % args.save_epoch == 0:
                    model.save(savepath)
                    for send in range(args.save_epoch):
                        monitor_epoch_end()
            monitor_batch_end()
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
    monitor_train_end()
