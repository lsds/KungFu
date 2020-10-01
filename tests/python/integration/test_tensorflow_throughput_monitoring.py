import argparse

import kungfu
import tensorflow as tf
from kungfu.tensorflow.ops import egress_rates, monitored_all_reduce
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_args():
    p = argparse.ArgumentParser(description='Test')
    p.add_argument('--max-step', type=int, default=60, help='max step')
    p.add_argument('--data-size', type=int, default=1024, help='max step')
    return p.parse_args()


def build_fake_train_op(args):
    xs = [tf.Variable(tf.ones((args.data_size, 1)))]
    ys = [monitored_all_reduce(x) for x in xs]
    return ys


def main():
    args = parse_args()
    train_op = build_fake_train_op(args)
    egress_rates_op = egress_rates()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for step in range(args.max_step):
            # BEGIN train
            vs = sess.run(train_op)
            print('step %d, result: %d' % (step, vs[0].sum()))
            # END train

            rates = sess.run(egress_rates_op)
            print(rates)


main()
