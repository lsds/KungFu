import argparse

import kungfu
import tensorflow as tf
from kungfu.python import detached
from kungfu.tensorflow.ops import all_reduce, resize
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_args():
    p = argparse.ArgumentParser(description='Test')
    p.add_argument('--max-step', type=int, default=60, help='max step')
    p.add_argument('--use-nccl', action='store_true', default=False, help='')
    return p.parse_args()


def build_fake_train_op(use_nccl):
    xs = [tf.Variable(tf.ones((2, 5)))]
    if use_nccl:
        from kungfu.tensorflow.ops import group_nccl_all_reduce
        ys = group_nccl_all_reduce(xs)
    else:
        from kungfu.tensorflow.ops import group_all_reduce
        ys = group_all_reduce(xs)
    return ys


def main():
    # step -> new_size
    fake_schedule = {
        10: 2,
        20: 3,
        40: 4,
        50: 1,
    }
    args = parse_args()
    gs = tf.train.get_or_create_global_step()
    sync_step_op = tf.assign(gs, all_reduce(gs, op='max'))
    inc_gs = tf.assign_add(gs, 1)
    new_size = tf.placeholder(dtype=tf.uint32)
    resize_op = resize(new_size)
    train_op = build_fake_train_op(args.use_nccl)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        need_sync = True
        while True:
            if need_sync:
                sess.run(sync_step_op)
                need_sync = False

            step = sess.run(gs)

            # BEGIN train
            vs = sess.run(train_op)
            print('step %d, result: %d' % (step, vs[0].sum()))
            # END train

            if step in fake_schedule:
                changed = sess.run(resize_op,
                                   feed_dict={new_size: fake_schedule[step]})
                if changed:
                    need_sync = True
                    if detached():
                        break
                else:
                    print('cluster not changed')
                assert changed

            next_gs = sess.run(inc_gs)
            print('finished %s' % (next_gs - 1))
            if next_gs >= args.max_step:
                break

    print('stopped')


main()
