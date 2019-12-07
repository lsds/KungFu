"""
Usage:
    python3 -m kungfu.info

"""

import sys

from kungfu.ext import show_cuda_version, show_nccl_version


def main(_):
    show_cuda_version()
    show_nccl_version()


if __name__ == "__main__":
    main(sys.argv)
