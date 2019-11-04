import argparse
import os
import sys
import timeit

import numpy as np
import tensorflow as tf
from tensorflow.keras import applications

FLAGS = None


def main(_):
    print('Model: %s' % FLAGS.model)
    print('Batch size: %d' % FLAGS.batch_size)

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # Set the GPU
        config = tf.ConfigProto()
        use_cuda = not FLAGS.no_cuda
        if use_cuda:
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list = os.environ[
                "CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            config.gpu_options.allow_growth = False
            config.gpu_options.visible_device_list = ''

        # Assigns ops to the local worker by default.
        device_name = '/job:worker/task:%d' % FLAGS.task_index
        print(device_name)
        with tf.device(
                tf.train.replica_device_setter(worker_device=device_name,
                                               cluster=cluster)):
            # Build model
            model = getattr(applications, FLAGS.model)(weights=None)

            data = tf.random_uniform([FLAGS.batch_size, 224, 224, 3])
            target = tf.random_uniform([FLAGS.batch_size, 1],
                                       minval=0,
                                       maxval=999,
                                       dtype=tf.int64)
            logits = model(data, training=True)
            loss = tf.losses.sparse_softmax_cross_entropy(target, logits)
            opt = tf.train.GradientDescentOptimizer(0.01)
            global_step = tf.contrib.framework.get_or_create_global_step()
            train_op = opt.minimize(loss, global_step=global_step)

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=FLAGS.num_iters)]

        # Benchmark
        print('Running benchmark...')

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        img_secs = []
        x = 0
        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=(FLAGS.task_index == 0),
                config=config,
                hooks=hooks) as mon_sess:

            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                time = timeit.timeit(lambda: mon_sess.run(train_op), number=1)
                img_sec = FLAGS.batch_size / time
                print('Iter #%d: %.1f img/sec per %s' %
                      (x, img_sec, device_name))
                img_secs.append(img_sec)
                x = x + 1

        # Results
        img_sec_mean = np.mean(img_secs)
        img_sec_conf = 1.96 * np.std(img_secs)
        print('Img/sec per %s: %.1f +-%.1f' %
              (device_name, img_sec_mean, img_sec_conf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument("--ps_hosts",
                        type=str,
                        default="",
                        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--worker_hosts",
                        type=str,
                        default="",
                        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--job_name",
                        type=str,
                        default="",
                        help="One of 'ps', 'worker'")
    # Flags for defining the tf.train.Server
    parser.add_argument("--task_index",
                        type=int,
                        default=0,
                        help="Index of task within the job")
    # Flags for training
    parser.add_argument('--model',
                        type=str,
                        default='ResNet50',
                        help='model to benchmark')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='input batch size')
    parser.add_argument('--num-iters',
                        type=int,
                        default=3000,
                        help='number of benchmark iterations')
    # Flags for GPU
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
