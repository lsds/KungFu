import tensorflow as tf
from kungfu.tensorflow.ops import (all_reduce, broadcast, counter,
                                   _get_init_cluster_version_id,
                                   step_based_schedule)
from kungfu.tensorflow.optimizers import ElasticSyncSGDOptimizer


def get_config():
    stage_sizes = [1, 2, 4, 8]
    step_per_stage = 3

    config = ','.join('%d:%d' % (size, step_per_stage) for size in stage_sizes)
    max_step = step_per_stage * len(stage_sizes)
    return config, max_step


config, max_step = get_config()
init_step = int(_get_init_cluster_version_id())


def build_optimizer():
    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return ElasticSyncSGDOptimizer(optimizer, config, init_step)


def build_ops():
    optimizer = build_optimizer()

    x = tf.Variable(1.0, tf.float32)
    y = x * x
    train_step = optimizer.minimize(y)

    sync_op = tf.assign(x, broadcast(x))
    init_op = tf.global_variables_initializer()

    return init_op, sync_op, train_step, y


print('init_step is %d' % (init_step))
init_op, sync_op, train_step, y = build_ops()

with tf.Session() as sess:
    sess.run(init_op)
    need_sync = True
    for i in range(init_step, max_step):
        if need_sync:
            sess.run(sync_op)
            need_sync = False

        print('step: %d' % (i))
        need_sync, keep = sess.run(train_step)
        v = sess.run(y)
        print('step %d, y=%f' % (i, v))
        if not keep:
            break
