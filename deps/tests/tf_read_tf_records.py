#!/usr/bin/env python3
import os
import glob


def read_lines(filename):
    return [line.strip() for line in open(filename)]


home = os.getenv('HOME')
prefix = os.path.join(home, 'mnt/tfrecords')
print(prefix)

f = 'progress-0/cluster-1/rank-000/list.txt'

lines = read_lines(os.path.join(prefix, f))

print(lines)

for l in lines:
    # TODO: read
    print(l)
