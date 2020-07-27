import torch

from .clib import all_reduce_op_map


def all_reduce_fn(x, op=None):
    if op is None:
        op = 'sum'
    y = x.new(x.shape)
    all_reduce_op_map[x.type()](x, y, x.type(), op)
    return y


def inplace_all_reduce_op(x, op=None):
    if op is None:
        op = 'sum'
    all_reduce_op_map[x.type()](x, x, x.type(), op)
