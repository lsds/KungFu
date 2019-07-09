#!/usr/bin/env python3

import tensorflow as tf
from kungfu.ops import save_variables, barrier, request, global_minimum_spanning_tree
from kungfu.internal import _get_num_peers, _get_other_ranks, _get_self_rank

np = _get_num_peers()
rank = _get_self_rank()
others = _get_other_ranks()


def show_mst(edges):
    return ', '.join('(%d,%d)' % (u, v) for [u, v] in edges)


def test_dynamic_topology():
    target = tf.Variable(tf.constant((rank + 1) % np))
    # targets = [tf.Variable(tf.constant(r)) for r in others]

    w = tf.Variable(tf.ones([8, 3, 3, 32]))
    b = tf.Variable(tf.ones([32]))
    variables = [w, b]

    requested_vars = [request(target, v.name, v)
                      for v in variables]  # FIXME: update target based on mst

    latencies = tf.Variable(tf.ones([np], dtype=tf.float32))

    with tf.control_dependencies([save_variables(variables)]):
        init = barrier()

    mst_edges = global_minimum_spanning_tree(latencies)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init)

        n = 3
        for step in range(n):
            print('step: %d' % (step))
            t = sess.run(target)
            print('target of %d is %d' % (rank, t))
            es = sess.run(mst_edges)
            print('mst edges: %s' % (show_mst(es)))
            sess.run(requested_vars)


test_dynamic_topology()
