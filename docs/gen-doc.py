#!/usr/bin/env python3

import os
import sys
from inspect import getmembers, isclass, isfunction

import kungfu
import kungfu.tensorflow.v1.ops
import kungfu.tensorflow.v1.optimizers


def list_classes(m):
    for _name, f in getmembers(m):
        if isclass(f):
            yield f


def list_functions(m):
    for _name, f in getmembers(m):
        if isfunction(f):
            yield f


def is_private(o):
    return o.__name__.startswith('_')


def gen_module_doc(m):
    for f in list_functions(m):
        if is_private(f):
            continue
        yield '.. autofunction:: %s.%s' % (m.__name__, f.__name__)

    for c in list_classes(m):
        if is_private(c):
            continue
        yield '.. autoclass:: %s.%s' % (m.__name__, c.__name__)


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
