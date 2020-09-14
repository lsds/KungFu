#!/usr/bin/env python3

import kungfu.torch as kf
import torch
from kungfu.python import current_cluster_size, current_rank


def test_all_reduce(device='cpu'):
    x = torch.ones([2, 3])
    x.to(device)
    y = kf.ops.collective.all_reduce_fn(x)
    assert (x.shape == y.shape)
    np = current_cluster_size()
    z = x * np
    assert z.equal(y)


def test_all_gather(device='cpu'):
    rank = current_rank()
    x = (torch.ones([2, 3]) * rank)
    x.to(device)
    y = kf.ops.collective.all_gather(x)
    z = []
    np = current_cluster_size()
    for i in range(np):
        z.append(torch.ones([2, 3]) * i)
    z = torch.stack(z)
    assert (z.equal(y))


def test_device(device):
    tests = [
        test_all_reduce,
        test_all_gather,
    ]
    for t in tests:
        print('running %s on %s' % (t, device))
        t(device)


test_device('cpu')
if torch.cuda.is_available():
    test_device('cuda:0')
