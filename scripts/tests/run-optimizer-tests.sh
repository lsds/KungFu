#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)
cd ../..
ROOT=$(pwd)

if [ -z "${PYTHON}" ]; then
    PYTHON=$(which python3)
fi

. ./scripts/utils/measure.sh

reset_go_mod() {
    echo 'module github.com/lsds/KungFu' >go.mod
    if [ -f go.sum ]; then
        rm go.sum
    fi
}

KUNGFU_RUN=${ROOT}/bin/kungfu-run

ensure_kungfu_run() {
    if [ ! -f ${KUNGFU_RUN} ]; then
        reset_go_mod
        GOBIN=$PWD/bin go install -v ./srcs/go/cmd/kungfu-run
    fi
}

ensure_kungfu_run

run_operator_tests() {
    local SCRIPT=$1
    local max_np=4
    for np in $(seq $max_np); do
        local hosts=127.0.0.1:$np
        $KUNGFU_RUN \
            -np $np \
            -H $hosts \
            ${PYTHON} \
            ${SCRIPT}
    done
}

run_adaptation_tests() {
    schedule='5:1,5:2,5:4,5:8,5:4,5:2,5:1'
    $KUNGFU_RUN \
        -H '127.0.0.1:8' \
        -np 1 \
        -w \
        ${PYTHON} \
        tests/python/test_optimizers.py \
        --test elastic-sgd \
        --schedule $schedule
}

measure run_operator_tests ${ROOT}/tests/python/test_optimizers.py
measure run_adaptation_tests
