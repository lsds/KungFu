import torch

from .clib import all_reduce_async_op_map, all_reduce_op_map, ops


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


def inplace_all_reduce_async_op(x, name, op=None):
    if op is None:
        op = 'sum'
    return all_reduce_async_op_map[x.type()](x, x, x.type(), op, name)


def wait_handle(handle):
    ops.wait_handle(handle)


def wait_all_handles(handles):
    ops.wait_all_handles(handles)
