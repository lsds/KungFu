import sys

import tensorflow as tf
from kungfu.tensorflow.v1.ops import (all_reduce, current_cluster_size,
                                      current_rank)


def show_info_example():
    rank = current_rank()
    np = current_cluster_size()
    print('rank=%d, np=%d' % (rank, np))


def all_reduce_example():
    pass


def main(args):
    show_info_example()
    all_reduce_example()


if __name__ == "__main__":
    main(sys.argv)
