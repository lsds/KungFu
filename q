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

ci() {
    git add -A
    git clean -fdx
    measure rebuild

    git clean -fdx
    measure reinstall
    which kungfu-run
}

ci

kungfu_run_flags() {
    # echo -q
    echo -allow-nvlink
    echo -logdir logs
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

# export NCCL_DEBUG=WARN
export NCCL_DEBUG=INFO
# kungfu_run -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method NCCL
# kungfu_run -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method NCCLv2
kungfu_run -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method NCCLv2+CPU
# kungfu_run -np 4 python3 benchmarks/system/benchmark_kungfu.py --batch-size 1 --kf-optimizer sync-sgd-nccl
# kungfu_run -np 4 python3 benchmarks/system/benchmark_kungfu.py --batch-size 1 --kf-optimizer sync-sgd-hierarchical-nccl

# kungfu_run -np 4 python3 benchmarks/system/benchmark_kungfu.py --batch-size 32 --kf-optimizer sync-sgd-nccl
# kungfu_run -np 4 python3 benchmarks/system/benchmark_kungfu.py --batch-size 32 --kf-optimizer sync-sgd-hierarchical-nccl
