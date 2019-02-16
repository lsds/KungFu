#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$(cd ../../.. && pwd)

if [ $(uname -s) = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
fi

single_train() {
    python3 ./pingpong.py
}

parallel_train() {
    local KUNGFU_PRUN=${ROOT}/bin/kungfu-prun
    if [ ! -f ${KUNGFU_PRUN} ]; then
        ${ROOT}/scripts/go-install.sh
    fi

    local checkpoint=pingpong.npz

    local total_batch_size=10
    local np=4
    local batch_size=$((total_batch_size / np))

    python3 ./pingpong.py --init=1 --checkpoint=$checkpoint
    local timeout=10m

    ${KUNGFU_PRUN} \
        -timeout $timeout \
        -np $np \
        python3 \
        ./pingpong.py \
        --checkpoint $checkpoint \
        --batch-size $batch_size
}

async() {
    $@ >out.log 2>err.log &
}

# single_train
# parallel_train
async parallel_train
