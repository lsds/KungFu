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
measure rebuild

git clean -fdx
measure reinstall
which kungfu-run

kungfu_run_flags() {
    echo -q
    echo -allow-nvlink
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

kungfu_run -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method NCCL+CPU
kungfu_run -np 4 python3 benchmarks/system/benchmark_kungfu.py --batch-size 1 --kf-optimizer sync-sgd-nccl
# kungfu_run -np 4 python3 benchmarks/system/benchmark_kungfu.py --batch-size 1 --kf-optimizer sync-sgd-hierarchical-nccl
