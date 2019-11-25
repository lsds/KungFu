import tensorflow as tf
from kungfu.tensorflow.ops import (counter, get_init_checkpoint, all_reduce,
                                   resize_cluster, step_based_schedule)


def get_config():
    stage_sizes = [1, 2, 4, 8, 4, 2, 1]
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
    return init_step, resize_op


init_step, step_op = build_ops()
x = tf.Variable(1, tf.int32)
y = all_reduce(x)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(init_step, max_step):
        print(i)
        v = sess.run(y)
        print('step %d, np=%d' % (i, v))

        keep = sess.run(step_op)  # must be called exactly once per step
        if not keep:
            break
