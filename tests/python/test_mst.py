#!/usr/bin/env python3

import tensorflow as tf
from kungfu.ops import save_variables, barrier, request, global_minimum_spanning_tree, get_neighbour_mask, round_robin, get_peer_latencies
from kungfu.internal import _get_num_peers, _get_other_ranks, _get_self_rank

np = _get_num_peers()
rank = _get_self_rank()
others = _get_other_ranks()


def show_mst(edges):
    return ', '.join('(%d,%d)' % (u, v) for [u, v] in edges)


def test_dynamic_topology():
    w = tf.Variable(tf.ones([8, 3, 3, 32]))
    b = tf.Variable(tf.ones([32]))
    variables = [w, b]
    with tf.control_dependencies([save_variables(variables)]):
        init_op = barrier()
    final_op = barrier()

    latencies = get_peer_latencies()
    mst_edges = global_minimum_spanning_tree(latencies)

    new_mask = get_neighbour_mask(mst_edges)
    neighbour_mask = tf.Variable(tf.constant([r != rank for r in range(np)]))
    updata_mask = tf.assign(neighbour_mask, new_mask)

    target = round_robin(neighbour_mask)
    requested_vars = [request(target, v.name, v)
                      for v in variables]  # FIXME: update target based on mst

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init_op)

        n_epoches = 3
        n_steps = 5

        for epoch in range(1, 1 + n_epoches):
            print('epoch %d begins on %d' % (epoch, rank))

            mask, es, _ = sess.run([new_mask, mst_edges,
                                    updata_mask])  # update topology
            print('mst edges: %s' % (show_mst(es)))
            print(mask)

            for step in range(1, 1 + n_steps):
                t, _ = sess.run([target, requested_vars])
                print('epoch: %d, step: %d, target of %d is %d' %
                      (epoch, step, rank, t))

            print('epoch %d finished on %d' % (epoch, rank))

        sess.run(final_op)


test_dynamic_topology()
