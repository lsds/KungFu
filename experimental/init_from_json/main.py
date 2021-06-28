import argparse

import tensorflow as tf
from kungfu.python import current_cluster_size, current_rank, init_from_config
from kungfu.tensorflow.ops import all_reduce


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    config = {
        'cluster': {
            'peers': [
                '127.0.0.1:10010',
                '127.0.0.1:10011',
            ],
        },
        'self': {
            'rank': args.rank,
        },
    }

    init_from_config(config)

    rank = current_rank()
    size = current_cluster_size()
    print('%d/%d' % (rank, size))
    x = tf.Variable(1 + int(rank), dtype=tf.int32)
    y = all_reduce(x)
    print(x, y)
    print('done')


main()
