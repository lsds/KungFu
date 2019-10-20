import argparse

import tensorflow as tf
from kungfu.ops import (broadcast, global_noise_scale, group_all_reduce,
                        group_nccl_all_reduce, peer_info)

from kungfu.optimizers.core import KungFuOptimizer, defuse, fuse


class ElasticSGDOptimizer(KungFuOptimizer):
    def __init__(self,
                 optimizer,
                 nccl=False,
                 nccl_fusion=True,
                 name=None,
                 use_locking=False):
        super(ElasticSGDOptimizer, self).__init__(optimizer,
                                                  name,
                                                  use_locking=use_locking)
        _rank, np = peer_info()
        # FIXME: use type of gradient
        self._num_workers = tf.cast(np, tf.float32)
        self._nccl = nccl
        self._nccl_fusion = nccl_fusion

    def apply_gradients(self, grads_and_vars, **kwargs):
        gradients, variables = list(zip(*grads_and_vars))

        if self._nccl:
            if self._nccl_fusion:
                fused_grad = fuse(gradients)
                summed_fused_gradients = group_nccl_all_reduce([fused_grad])
                summed_gradients = defuse(summed_fused_gradients[0],
                                          [g.shape for g in gradients])
            else:
                summed_gradients = group_nccl_all_reduce(gradients)
        else:
            summed_gradients = group_all_reduce(gradients)

        reduced_grads = [g / self._num_workers for g in summed_gradients]
        reduced_grads_and_vars = zip(reduced_grads, variables)
        return self._optimizer.apply_gradients(reduced_grads_and_vars,
                                               **kwargs)

    def distributed_initializer(self):
        ops = [tf.assign(v, broadcast(v)) for v in self.variables()]
        return tf.group(ops)


def parse_schedule(config):
    schedule = []
    t = 0
    for p in config.split(','):
        kv = p.split(':')
        n, val = (int(kv[0]), int(kv[1]))
        schedule.append((t, t + n, val))
        t += n
    return schedule, t


def get_cluster_size(i, sch, old):
    for s, e, n in sch:
        if s <= i and i < e:
            return n
    print('[W] not scheduled for %d' % (i))
    return old


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
    # from kungfu.optimizers import ElasticSGDOptimizer
    from kungfu.ops.adapt import get_init_checkpoint, resize_cluster
    from kungfu.ops import current_cluster_size

    ckpt = tf.placeholder(tf.string)
    new_size = tf.placeholder(tf.int32)
    resize_op = resize_cluster(ckpt, new_size)

    schedule, max_step = parse_schedule(args.schedule)
    print(schedule)

    def restore(checkpoint):
        return int(checkpoint)

    init_step = restore(get_init_checkpoint())
    np = current_cluster_size()
    init_np = get_cluster_size(init_step, schedule, np)
    if np != init_np:
        print(
            '[W] init cluster size (np=%d) is not consistent with schedule (np=%d)'
            % (np, init_np))

    x = tf.Variable(tf.ones([], tf.float32))
    y = x * x
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = ElasticSGDOptimizer(optimizer)
    train_op = optimizer.minimize(y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(init_step, max_step):
            print('step=%d' % (step))
            sess.run(train_op)

            next_step = step + 1
            if next_step < max_step:
                new_np = get_cluster_size(next_step, schedule, np)
                if new_np != np:
                    keep = sess.run(resize_op,
                                    feed_dict={
                                        ckpt: str(next_step),
                                        new_size: new_np,
                                    })
                    np = new_np
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
