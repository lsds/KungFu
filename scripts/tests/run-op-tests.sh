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
if [ $(uname -s) = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$(${PYTHON} -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
fi

run_operator_tests() {
    local SCRIPT=$1
    for np in $(seq 4); do
        local hosts=127.0.0.1:$np
        $KUNGFU_RUN \
            -np $np \
            -H $hosts \
            ${PYTHON} \
            ${SCRIPT}
    done
}

measure run_operator_tests ${ROOT}/tests/python/test_operators.py
measure run_operator_tests ${ROOT}/tests/python/test_save_variables.py
