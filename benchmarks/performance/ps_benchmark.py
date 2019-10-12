import argparse
import sys
import os

import tensorflow as tf
from tensorflow.keras import applications

FLAGS = None


def main(_):
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
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            config.gpu_options.allow_growth = False
            config.gpu_options.visible_device_list = ''

        # Assigns ops to the local worker by default.
        # device_str = '/job:worker/task:%d/gpu:0' % FLAGS.task_index if use_cuda else '/job:worker/task:%d' % FLAGS.task_index
        # print(device_str)
        with tf.device(
                tf.train.replica_device_setter(
                    worker_device='/job:worker/task:%d' % FLAGS.task_index,
                    cluster=cluster)):
            # Start
            # Set up standard model.
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
            # End

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=FLAGS.num_iters)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=(FLAGS.task_index == 0),
                hooks=hooks,
                config=config) as mon_sess:
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                print("Run once")
                mon_sess.run(train_op)


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
                        default=32,
                        help='input batch size')
    parser.add_argument('--num-iters',
                        type=int,
                        default=1000,
                        help='number of benchmark iterations')
    # Flags for GPU
    parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
