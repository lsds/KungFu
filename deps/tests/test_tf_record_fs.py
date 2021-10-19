#!/usr/bin/env python3
"""
Check the integrity of TFRecord filesystem.
"""

import os
import glob


def read_int_file(filename):
    return int(open(filename).read())


def read_lines(filename):
    return [line.strip() for line in open(filename)]


home = os.getenv('HOME')
mount_point = os.path.join(home, 'mnt/tfrecords')
print(mount_point)


def all_the_same(lst):
    for x in lst:
        if x != lst[0]:
            return False

    return True


def test_cluster(progress, size):
    rel_path = 'progress-%d/cluster-of-%d' % (progress, size)
    prefix = os.path.join(mount_point, rel_path)
    total = read_int_file(mount_point + '/total.txt')
    print('total: %d' % (total))

    global_batch_size = read_int_file(prefix + '/global-batch-size.txt')
    print('global_batch_size: %d' % (global_batch_size))

    used_sampled = 0
    peer_steps = []
    bs_files = glob.glob(prefix + '/**/batch-sizes.txt', recursive=True)
    for filename in bs_files:
        for line in open(filename):
            parts = line.strip().split(' ')
            bs, cnt = [int(s) for s in parts]
            print('batch size %d for %d steps' % (bs, cnt))
            used_sampled += bs * cnt
            peer_steps.append(cnt)

    assert all_the_same(peer_steps), 'inconsistent steps'

    cluster_dropped = total - used_sampled - progress
    print('cluster dropped %s' % (cluster_dropped))

    drop_files = glob.glob(prefix + '/**/dropped.txt', recursive=True)
    for filename in drop_files:
        dropped = read_int_file(filename)
        print('dropped: %d' % (dropped))
        assert dropped == cluster_dropped, 'inconsistent drop'


def main():
    stage_files = glob.glob(mount_point + '/progress-*-cluster-of-*.txt')

    for f in stage_files:
        name = os.path.basename(f)  #progress-x-cluster-of-y.txt
        name = name.split('.')[0]
        parts = name.split('-')
        progress = int(parts[1])
        cluster_size = int(parts[4])
        print('test %s' % (name))
        test_cluster(progress, cluster_size)

    print('Check OK. TFRecord filesystem is consistent.')


main()
