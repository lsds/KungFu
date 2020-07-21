#!/bin/sh
set -e

. ./scripts/utils/measure.sh

cfg_flags() {
    echo --build-tensorflow-ops
    echo --enable-nccl
}

rebuild() {
    ./configure $(cfg_flags)
    make -j8
}

reinstall() {
    KUNGFU_ENABLE_NCCL=1 pip3 install --no-index -U .
}
git add -A
git clean -fdx
# measure rebuild

git clean -fdx
measure reinstall
which kungfu-run

kungfu-run -q -np 4 -allow-nvlink python3 -m kungfu.tensorflow.v1.benchmarks --method NCCL+CPU
