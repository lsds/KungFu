#!/usr/bin/env python3

import os
import sys
from inspect import getmembers, isclass, isfunction

import kungfu
import kungfu.tensorflow.v1.ops
import kungfu.tensorflow.v1.optimizers


def is_private(o):
    return o.__name__.startswith('_')


def list_members(m, pred, whitelist):
    for _name, f in getmembers(m):
        if not pred(f):
            continue
        if is_private(f):
            continue
        if not whitelist or _name in whitelist:
            yield f


def title(name, lvl):
    levels = ['=', '-', '~']
    return '%s\n%s\n' % (name, levels[lvl] * len(name))


def gen_module_doc(m, whitelist):
    yield title('module %s' % (m.__name__), 1)

    functions = list(list_members(m, isfunction, whitelist))
    if functions:
        yield title('functions', 2)
        for f in functions:
            yield '.. autofunction:: %s.%s' % (m.__name__, f.__name__)

    classes = list(list_members(m, isclass, whitelist))
    if classes:
        yield title('classes', 2)
        for c in classes:
            yield '.. autoclass:: %s.%s' % (m.__name__, c.__name__)

    yield ''


modules = [
    (kungfu, []),
    (kungfu.tensorflow.v1.ops, [
        'barrier',
        'group_all_reduce',
        'group_nccl_all_reduce',
    ]),
    (kungfu.tensorflow.v1.optimizers, []),
]


def gen_doc(modules):
    yield title('KungFu distributed machine learning framework', 1)

    for (m, whitelist) in modules:
        for line in gen_module_doc(m, whitelist):
            yield line


def main(args):
    output = 'index.rst'
    with open(output, 'w') as f:
        for line in gen_doc(modules):
            f.write(line + '\n')


main(sys.argv)
