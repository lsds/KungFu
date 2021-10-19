#!/usr/bin/env python3
import os
import glob
import time

import mindspore._c_dataengine as cde
import mindspore.dataset.engine as de


def measure(f, name):
    t0 = time.time()
    f()
    d = time.time() - t0
    print('%s took %.3fs' % (name, d))


def read_lines(filename):
    return [line.strip() for line in open(filename)]


def get_origin_files():
    return list(sorted(glob.glob('/data/imagenet/records/train*')))


def get_indexed_files():
    home = os.getenv('HOME')
    prefix = os.path.join(home, 'mnt/tfrecords')
    print('prefix: %s' % (prefix))

    list_files = glob.glob(prefix + '/**/list.txt', recursive=True)
    lines = read_lines(os.path.join(prefix, list_files[0]))

    filenames = [os.path.join(prefix, l[1:]) for l in lines]

    return filenames


def main():
    # filenames = get_origin_files()  # main took 611.879s
    filenames = get_indexed_files()  # main took 1112.984s
    print('got %d TFRecord files' % (len(filenames)))
    # for f in filenames:
    #     print(f)

    t0 = time.time()
    ds = de.TFRecordDataset(dataset_files=filenames)
    size = ds.get_dataset_size()
    dur = time.time() - t0
    print('%d samples in total, took %.3fs' % (size, dur))

    t0 = time.time()

    # ds = ds.batch(10)

    for step, item in enumerate(ds):
        # print(step)
        if step % 1000 == 0:
            print('step: %d' % (step))
            for i, t in enumerate(item):
                print('[{}] {}{}'.format(i, t.dtype, t.shape))

            # break

            dur = time.time() - t0
            print('loop took %.3fs' % (dur))
            t0 = time.time()

    print('main END')


# main()
measure(main, 'main')
