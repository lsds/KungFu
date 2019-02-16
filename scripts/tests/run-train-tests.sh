#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)
cd ../..
ROOT=$(pwd)

. ./scripts/utils/measure.sh

export KUNGFU_CONFIG_LOG_CONFIG_VARS=true
KUNGFU_PRUN=${ROOT}/bin/kungfu-prun
if [ ! -f ${KUNGFU_PRUN} ]; then
    ${ROOT}/scripts/go-install.sh
fi

if [ $(uname -s) = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
fi

SCRIPT=${ROOT}/srcs/python/tests/test_mnist_slp.py

epochs=2

run_single_train_test() {
    local total_batch_size=$1
    python3 ${SCRIPT} \
        --no-kungfu=1 \
        --n-epochs $epochs \
        --batch-size $total_batch_size

}

run_parallel_train_test() {
    local total_batch_size=$1
    local np=$2

    local hosts=127.0.0.1:$np
    local timeout=$((np * 20))s
    local batch_size=$((total_batch_size / np))

    ${KUNGFU_PRUN} \
        -timeout $timeout \
        -np $np \
        -H $hosts \
        python3 \
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
    measure run_train_tests 6000 1 4 6
    measure run_train_tests 600 1 3 4
    measure run_train_tests 500 1 2 5
    measure run_train_tests 50 1 2 5
}

measure run_all_test
