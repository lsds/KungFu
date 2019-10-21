import argparse

import tensorflow as tf
from kungfu.ops import all_reduce, broadcast, group_all_reduce, peer_info, current_cluster_size

from kungfu.optimizers.core import KungFuOptimizer, defuse, fuse


class ElasticSGDOptimizer(KungFuOptimizer):
    def __init__(self, optimizer, name=None, use_locking=False):
        super(ElasticSGDOptimizer, self).__init__(optimizer,
                                                  name,
                                                  use_locking=use_locking)
        _rank, np = peer_info()
        # FIXME: use type of gradient
        self._num_workers = tf.cast(np, tf.float32)
        self._step = tf.Variable(0, trainable=False, dtype=tf.int32)

    def _broadcast_variables(self, variables, step):
        ops = [tf.assign(v, broadcast(v)) for v in variables]
        print_op = tf.print('Broadcast variables at the step:', step)
        with tf.control_dependencies([print_op]):
            return tf.group(ops)

    def compute_gradients(self, *args, **kwargs):
        broadcast_cond_op = tf.cond(
            tf.equal(self._step, 0),
            lambda: self._broadcast_variables(self.variables(), self._step),
            lambda: tf.no_op())
        with tf.control_dependencies([broadcast_cond_op]):
            with tf.control_dependencies([tf.assign_add(self._step, 1)]):
                return self._optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, **kwargs):
        gradients, variables = list(zip(*grads_and_vars))
        summed_gradients = group_all_reduce(gradients)
        reduced_grads = [g / self._num_workers for g in summed_gradients]
        reduced_grads_and_vars = zip(reduced_grads, variables)
        return self._optimizer.apply_gradients(reduced_grads_and_vars,
                                               **kwargs)


def test_sync_sgd(args):
    from kungfu.optimizers import SyncSGDOptimizer
    x = tf.Variable(tf.ones([], tf.float32))
    y = x * x
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = SyncSGDOptimizer(optimizer)
    train_op = optimizer.minimize(y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            sess.run(train_op)
        # FIXME: check values


def test_elastic_sgd(args):
    from elastic_scheduler import ElasticScheduler

    elastic = ElasticScheduler(args.schedule)

    one = tf.Variable(tf.ones([], tf.int32))
    np = all_reduce(one)

    def build_train_op():
        x = tf.Variable(tf.ones([], tf.float32))
        y = x * x
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        # optimizer = ElasticSGDOptimizer(optimizer)
        return optimizer.minimize(y)

    train_op = build_train_op()
    train_op, elastic_op = elastic(train_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for stage in range(elastic.init_stage, elastic.max_stage):
            v = sess.run(np)
            print('step: %d, result: %d' % (stage, v))
            _, keep = sess.run([train_op, elastic_op])
            if not keep:
                break


all_tests = {
    'sync-sgd': test_sync_sgd,
    'elastic-sgd': test_elastic_sgd,
    # TODO: more tests
}


def parse_args():
    parser = argparse.ArgumentParser(description='Tests.')
    parser.add_argument('--test', type=str, default='sync-sgd', help='')
    parser.add_argument('--schedule', type=str, default='', help='')
    return parser.parse_args()


def main():
    args = parse_args()
    t = all_tests[args.test]
    t(args)


main()
