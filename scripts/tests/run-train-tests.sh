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
        ./configure --no-tests --build-tools && make
    fi
}

ensure_kungfu_run
export KUNGFU_CONFIG_LOG_CONFIG_VARS=true

SCRIPT=${ROOT}/tests/python/integration/test_mnist_slp.py

epochs=2

run_single_train_test() {
    local total_batch_size=$1
    ${PYTHON} ${SCRIPT} \
        --no-kungfu=1 \
        --n-epochs $epochs \
        --batch-size $total_batch_size

}

run_parallel_train_test() {
    local total_batch_size=$1
    local np=$2

    local hosts=127.0.0.1:$np
    local timeout=$((np * 8))s
    local batch_size=$((total_batch_size / np))

    ${KUNGFU_RUN} \
        -timeout $timeout \
        -np $np \
        -H $hosts \
        ${PYTHON} \
        ${SCRIPT} \
        --n-epochs $epochs \
        --batch-size $batch_size
}

run_train_tests() {
    local total_batch_size=$1
    shift

    measure run_single_train_test $total_batch_size
    for np in $@; do
        measure run_parallel_train_test $total_batch_size $np
    done
}

run_all_test() {
    measure run_train_tests 6000 1 4
    measure run_train_tests 600 1 3 4
    measure run_train_tests 500 1 2 4
    measure run_train_tests 50 1 2
}

measure run_all_test
