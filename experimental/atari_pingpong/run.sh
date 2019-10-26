#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$(cd ../../.. && pwd)

single_train() {
    export PYTHONUNBUFFERED=1
    python3 ./pingpong.py
}

parallel_train() {
    local KUNGFU_RUN=${ROOT}/bin/kungfu-run
    if [ ! -f ${KUNGFU_RUN} ]; then
        ${ROOT}/scripts/go-install.sh
    fi

    local checkpoint=pingpong.npz

    local total_batch_size=10
    local np=4
    local batch_size=$((total_batch_size / np))

    python3 ./pingpong.py --init=1 --checkpoint=$checkpoint
    local timeout=10m

    ${KUNGFU_RUN} \
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

ASYNC=async

$ASYNC single_train
# $ASYNC parallel_train
