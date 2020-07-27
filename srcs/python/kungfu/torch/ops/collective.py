import torch

from .clib import ops


def all_reduce_fn(x, op=None):
    if op is None:
        op = 'sum'
    y = x.new(x.shape)
    ops.all_reduce(x, y, x.type(), op)
    return y


def inplace_all_reduce_op(x, op=None):
    if op is None:
        op = 'sum'
    ops.all_reduce(x, x, x.type(), op)
