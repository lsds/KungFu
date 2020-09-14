#!/usr/bin/env python3

import kungfu.torch as kf
import torch
from kungfu.python import current_cluster_size, current_rank


def test_all_reduce():
    x = torch.ones([2, 3])
    y = kf.ops.collective.all_reduce_fn(x)
    assert (x.shape == y.shape)
    np = current_cluster_size()
    z = x * np
    assert z.equal(y)


def test_all_gather():
    rank = current_rank()
    x = torch.ones([2, 3]) * rank
    y = kf.ops.collective.all_gather(x)
    z = []
    np = current_cluster_size()
    for i in range(np):
        z.append(torch.ones([2, 3]) * i)
    z = torch.stack(z)
    assert (z.equal(y))


def test_all():
    tests = [
        test_all_reduce,
        test_all_gather,
    ]
    for t in tests:
        print('running %s' % t)
        t()


test_all()
