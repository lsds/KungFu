#!/usr/bin/env python3

import kungfu.torch as kf
import torch
from kungfu.python import current_cluster_size


def test_all_reduce():
    x = torch.ones([2, 2])
    y = kf.ops.collective.all_reduce_fn(x)
    assert (x.shape == y.shape)
    # TODO: check value of y
    # np = current_cluster_size()
    # z = x * np
    # assert (y == z)


def test_all_gather():
    x = torch.ones([2, 2])
    # print(x)

    y = kf.ops.collective.all_gather(x)
    # TODO: check shape and value of y


def test_all():
    tests = [
        test_all_reduce,
        test_all_gather,
    ]
    for t in tests:
        print('running %s' % t)
        t()


test_all()
