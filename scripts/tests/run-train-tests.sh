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

run_train_test() {

    local total_batch_size=6000 # train 60k images in 10 steps
    local np=$1

    local timeout=20s

    local epochs=2
    local batch_size=$((total_batch_size / np))

    ${KUNGFU_PRUN} \
        -timeout $timeout \
        -np $np \
        python3 \
        ${ROOT}/srcs/python/kungfu_tests/test_mnist_slp.py \
        --n-epochs $epochs \
        --batch-size $batch_size
}

measure run_train_test 1
measure run_train_test 2
measure run_train_test 3
measure run_train_test 4
