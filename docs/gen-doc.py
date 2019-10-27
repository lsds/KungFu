#!/usr/bin/env python3

import os
import sys
from inspect import getmembers, isclass, isfunction

import kungfu
import kungfu.tensorflow.v1.ops
import kungfu.tensorflow.v1.optimizers


def is_private(o):
    return o.__name__.startswith('_')


def list_members(m, pred):
    for _name, f in getmembers(m):
        if not pred(f):
            continue
        if is_private(f):
            continue
        yield f


def title(name, lvl):
    levels = ['=', '-', '~']
    return '%s\n%s\n' % (name, levels[lvl] * len(name))


def gen_module_doc(m):
    yield title('module %s' % (m.__name__), 1)
    functions = list(list_members(m, isfunction))
    if functions:
        yield title('functions', 2)
        for f in functions:
            yield '.. autofunction:: %s.%s' % (m.__name__, f.__name__)

    classes = list(list_members(m, isclass))
    if classes:
        yield title('classes', 2)
        for c in classes:
            yield '.. autoclass:: %s.%s' % (m.__name__, c.__name__)

    yield ''


modules = [
    (kungfu, ),
    (kungfu.tensorflow.v1.ops, ),
    (kungfu.tensorflow.v1.optimizers, ),
]


def main(args):
    output = 'index.rst'
    with open(output, 'w') as f:
        for (m, ) in modules:
            for line in gen_module_doc(m):
                f.write(line + '\n')


main(sys.argv)
