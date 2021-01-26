#!/usr/bin/env python3

import torch
from kungfu.cmd import launch_multiprocess


def worker(rank):
    import kungfu.torch as kf
    from kungfu.python import current_cluster_size, current_rank
    print('rank=%d' % (rank))
    print('kungfu rank: %d, size %d' %
          (current_rank(), current_cluster_size()))
    x = torch.ones([]) * int(current_rank())
    print(x)
    y = kf.ops.collective.all_reduce_fn(x)
    print(y)


def main():
    np = 4
    launch_multiprocess(worker, np)


main()
