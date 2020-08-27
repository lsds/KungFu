"""
Usage:
    python3 -m kungfu.info

"""

import sys

from kungfu.python import show_cuda_version, show_nccl_version


def _show_tensorflow_info():
    try:
        import tensorflow as tf
        print('Tensorflow Version: %s' % (tf.__version__))
    except:
        print('Tensorflow is NOT installed')


def main(_):
    show_cuda_version()
    show_nccl_version()
    _show_tensorflow_info()


if __name__ == "__main__":
    main(sys.argv)
