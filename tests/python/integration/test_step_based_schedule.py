import tensorflow as tf
from kungfu.tensorflow.ops import (counter, all_reduce, broadcast,
                                   resize_cluster, step_based_schedule)


def get_config():
    stage_sizes = [1, 2, 4, 8, 4, 2, 1, 2, 4, 8, 4, 2, 1]
    step_per_stage = 3

    config = ','.join('%d:%d' % (size, step_per_stage) for size in stage_sizes)
    max_step = step_per_stage * len(stage_sizes)
    return config, max_step


config, max_step = get_config()


def build_ops():
    step_place = tf.placeholder(dtype=tf.int32, shape=())
    schedule = step_based_schedule(config, step_place)
    resize_op = resize_cluster(schedule)
    return step_place, resize_op


step_place, resize_op = build_ops()
sync_step_op = all_reduce(step_place, op='max')
x = tf.Variable(1, tf.int32)
y = all_reduce(x)

sync_state_op = tf.assign(x, broadcast(x))
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    need_sync = True
    i = 0
    while i < max_step:
        if need_sync:
            new_step = sess.run(sync_step_op, feed_dict={step_place: i})
            print('sync step: %d -> %d' % (i, new_step))
            i = new_step
            sess.run(sync_state_op)

        print(i)
        v = sess.run(y)
        print('step %d, np=%d' % (i, v))

        # must be called exactly once per step
        need_sync, keep = sess.run(resize_op, feed_dict={step_place: i})
        if not keep:
            break
        i += 1
