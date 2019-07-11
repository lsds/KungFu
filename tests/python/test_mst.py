#!/usr/bin/env python3

import tensorflow as tf
from kungfu.ops import save_variables, barrier, request, global_minimum_spanning_tree, get_neighbour_mask, round_robin
from kungfu.internal import _get_num_peers, _get_other_ranks, _get_self_rank

np = _get_num_peers()
rank = _get_self_rank()
others = _get_other_ranks()


def show_mst(edges):
    return ', '.join('(%d,%d)' % (u, v) for [u, v] in edges)


def gen_fake_latency_matrix(n):
    m = []
    for i in range(n):
        m.append([abs(i - j) for j in range(n)])
    return m


def measure_latency():
    fake_latency_mat = gen_fake_latency_matrix(np)
    latency_from_me = fake_latency_mat[rank]
    latencies = tf.Variable(tf.constant(latency_from_me, dtype=tf.float32))
    return latencies


def test_dynamic_topology():
    w = tf.Variable(tf.ones([8, 3, 3, 32]))
    b = tf.Variable(tf.ones([32]))
    variables = [w, b]
    with tf.control_dependencies([save_variables(variables)]):
        init = barrier()

    latencies = measure_latency()
    mst_edges = global_minimum_spanning_tree(latencies)

    new_mask = get_neighbour_mask(mst_edges)
    neighbour_mask = tf.Variable(tf.constant([r != rank for r in range(np)]))
    updata_mask = tf.assign(neighbour_mask, new_mask)

    target = round_robin(neighbour_mask)
    requested_vars = [request(target, v.name, v)
                      for v in variables]  # FIXME: update target based on mst

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init)

        n_epoches = 2
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


test_dynamic_topology()
