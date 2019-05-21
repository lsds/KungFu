import time
import tensorflow as tf
import numpy as np

import kungfu as kf
from kungfu import ops
from kungfu.ops import _tensor_size, _bin_pack

from resnet50 import grad_sizes

import os
self_rank = int(os.getenv('KUNGFU_TEST_SELF_RANK'))


def gen_test_tensors(sizes):
    ts = []
    for size in sizes:
        ts.append(tf.Variable(tf.ones(shape=(size, ), dtype=tf.float32)))
    return ts


ts = gen_test_tensors(grad_sizes)
total_size = sum([_tensor_size(t) for t in ts])
budget = int(0.1 * total_size)
indexes, num_partitions = _bin_pack(
    dict((t.name, _tensor_size(t)) for t in ts), budget)
groups = [[] for _ in range(num_partitions)]
for t in ts:
    groups[indexes[t.name]].append(t)

print("Group zero contains %d tensors" % len(groups[0]))
for t in groups[0]:
    print("Tensor name: %s" % t.name)

destination = tf.constant(1)

cond_ops = []
for t in groups[0]:
    op = tf.cond(tf.not_equal(
        destination,
        self_rank), lambda: ops.send_to(destination, t), lambda: tf.no_op())
    cond_ops.append(op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(cond_ops)
    time.sleep(2)
