import torch
from kungfu.python import current_cluster_size

from .clib import (all_gather_op_map, all_reduce_async_op_map,
                   all_reduce_op_map, broadcast_async_op_map, ops)


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


def inplace_broadcast_async_op(x, name):
    return broadcast_async_op_map[x.type()](x, x, x.type(), name)


def wait_handle(handle):
    ops.wait_handle(handle)


def wait_all_handles(handles):
    ops.wait_all_handles(handles)


def broadcast_parameters(state_dict):
    handles = []
    for name, value in state_dict.items():
        h = inplace_broadcast_async_op(value, name)
        handles.append(h)
    wait_all_handles(handles)


def all_gather(x):
    np = current_cluster_size()
    y = x.new(torch.Size([np] + list(x.shape)))
    all_gather_op_map[x.type()](x, y, x.type())
    return y
