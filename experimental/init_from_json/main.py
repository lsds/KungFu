import argparse

from kungfu.python import init_from_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    config = {
        'cluster': {
            'worker': [
                '127.0.0.1:10010',
                '127.0.0.1:10011',
            ],
        },
        'task': {
            'index': args.index,
        },
    }

    init_from_config(config)
    print('done')


main()
