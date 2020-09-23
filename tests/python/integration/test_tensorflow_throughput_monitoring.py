import argparse

import kungfu
import tensorflow as tf
from kungfu.tensorflow.ops import monitored_all_reduce


def parse_args():
    p = argparse.ArgumentParser(description='Test')
    p.add_argument('--max-step', type=int, default=60, help='max step')
    return p.parse_args()


def build_fake_train_op():
    xs = [tf.Variable(tf.ones((2, 5)))]
    ys = [monitored_all_reduce(x) for x in xs]
    return ys


def main():
    args = parse_args()
    train_op = build_fake_train_op()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for step in range(args.max_step):
            # BEGIN train
            vs = sess.run(train_op)
            print('step %d, result: %d' % (step, vs[0].sum()))
            # END train
            # TODO: get monitoring matrics


main()
