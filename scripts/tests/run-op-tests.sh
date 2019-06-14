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

KUNGFU_PRUN=${ROOT}/bin/kungfu-prun

SCRIPT=${ROOT}/tests/python/test_operators.py

if [ $(uname -s) = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$(${PYTHON} -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
fi

run_operator_tests() {
    for np in $(seq 4); do
        local hosts=127.0.0.1:$np
        $KUNGFU_PRUN \
            -np $np \
            -H $hosts \
            ${PYTHON} \
            ${SCRIPT}
    done
}

measure run_operator_tests
