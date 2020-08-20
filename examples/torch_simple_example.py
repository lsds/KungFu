#!/usr/bin/env python3

import kungfu.torch as kf
import torch

x = torch.ones([2, 2])
print(x)

y = kf.ops.collective.all_reduce_fn(x)
print(y)
