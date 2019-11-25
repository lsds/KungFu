import tensorflow as tf
from kungfu.tensorflow.ops import (counter, get_init_checkpoint,
                                   resize_cluster, step_based_schedule)


def get_config():
    stage_sizes = [1, 2, 4, 8]
    step_per_stage = 3

    config = ','.join('%d:%d' % (size, step_per_stage) for size in stage_sizes)
    max_step = step_per_stage * len(stage_sizes)
    return config, max_step


config, max_step = get_config()


def build_ops():
    init_step = int(get_init_checkpoint())
    print('init_step is %d' % (init_step))

    step = counter(init_step)
    schedule = step_based_schedule(config, step)
    ckpt_tensor = tf.as_string(step + 1)
    resize_op = resize_cluster(ckpt_tensor, schedule)
    init = tf.global_variables_initializer()
    return init_step, init, resize_op


init_step, init_op, step_op = build_ops()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(init_step, max_step):
        print(i)
        sess.run(step_op)
