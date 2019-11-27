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

. ./scripts/tests/common.sh

run_adaptation_tests() {
    schedule='5:1,5:2,5:4,5:8,5:4,5:2,5:1'
    $KUNGFU_RUN \
        -H '127.0.0.1:8' \
        -np 1 \
        -w \
        ${PYTHON} \
        tests/python/integration/test_optimizers.py \
        --test elastic-sgd \
        --schedule $schedule
}

measure run_tests 4 ${PYTHON} ${ROOT}/tests/python/integration/test_optimizers.py
# measure run_adaptation_tests
