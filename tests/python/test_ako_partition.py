import unittest

import os
import argparse

import itertools as it
from functools import reduce

import kungfu as kf
import tensorflow as tf

import random

def make_list_of_size(n):
    return list(it.repeat(0, n))

def tensor_size(t):
    return t.shape.num_elements() * t.dtype.size

def build_size_frequency_map(buckets):
    buckets_sizes = [list(map(lambda gv: tensor_size(gv[0]), bucket)) for bucket  in buckets]
    reduced_buckets = [sum(b) for b in buckets_sizes]
    frequency = dict()
    for size in reduced_buckets:
        if size not in frequency:
            frequency[size] = 1
        else:
            frequency[size] += 1
    return frequency

class TestSum(unittest.TestCase):
    def test_sum(self):
        opt = tf.train.GradientDescentOptimizer(0.01)
        opt = kf.SyncSGDOptimizer(opt, strategy='ako')

        grads_and_vars = [[tf.constant(make_list_of_size(n)), None] for n in range(1, 10)]
        k = 3
        buckets = [None for i in range(k)]
        buckets[0] = [(tf.constant(make_list_of_size(n)), None) for n in range(1, 6)]
        buckets[1] = [(tf.constant(make_list_of_size(n)), None) for n in range(6, 8)]
        buckets[2] = [(tf.constant(make_list_of_size(n)), None) for n in range(8, 10)]
        actual_buckets = opt.partition_gradients(grads_and_vars, k)
        self.assertEqual(len(actual_buckets), k, "Wrong bucket size")

        frequency_actual = build_size_frequency_map(actual_buckets)
        frequency_expected = build_size_frequency_map(buckets)
        self.assertEqual(frequency_actual, frequency_expected, "Wrong partitions")

if __name__ == '__main__':
    unittest.main()
