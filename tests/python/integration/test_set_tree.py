import time

import numpy as np
import tensorflow as tf


def gen_binary_tree(n, permu):
    father = [i for i in range(n)]
    father[permu[0]] = permu[0]
    for i in range(n):
        j, k = i * 2 + 1, i * 2 + 2
        if j < n:
            father[permu[j]] = permu[i]
        if k < n:
            father[permu[k]] = permu[i]
    return father


def gen_tree(n):
    permu = np.random.permutation(n)
    tree = gen_binary_tree(n, permu)
    return tree


def test_set_tree(steps, warmup_steps=10):
    from kungfu.python import current_cluster_size
    from kungfu.tensorflow.ops import all_reduce, broadcast, set_tree

    n = current_cluster_size()

    tree_place = tf.placeholder(dtype=tf.int32, shape=(n, ))
    set_tree_op = set_tree(broadcast(tree_place))

    magic = 32
    x = tf.Variable(list(range(magic)), dtype=tf.int32)
    y = all_reduce(x)

    init = tf.global_variables_initializer()

    durations = []
    with tf.Session() as sess:
        sess.run(init)
        from kungfu._utils import one_based_range
        for step in one_based_range(steps + warmup_steps):
            v = sess.run(y)
            assert (v.sum() == n * magic * (magic - 1) / 2)
            # print(v)

            tree = gen_tree(n)
            t0 = time.time()
            sess.run(set_tree_op, feed_dict={
                tree_place: tree,
            })
            dur = time.time() - t0

            if step > warmup_steps:
                durations.append(dur)

    ds = np.array([d * 1000 for d in durations])
    from kungfu._utils import show_duration
    print(
        'test set_tree OK for %d times among %d peers, took ~ %f <- [%f, %f] (ms)'
        % (len(ds), n, ds.mean(), ds.min(), ds.max()))


test_set_tree(32)
