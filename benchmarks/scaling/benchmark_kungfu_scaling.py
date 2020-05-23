import argparse

import tensorflow


def parse_args():
    p = argparse.ArgumentParser(
        description='TensorFlow Synthetic Benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--train-steps',
                   type=int,
                   default=10,
                   help='number of batches per benchmark iteration')

    p.add_argument('--model',
                   type=str,
                   default='ResNet50',
                   help='model to benchmark')
    return p.parse_args()


def main():
    args = parse_args()


main()
