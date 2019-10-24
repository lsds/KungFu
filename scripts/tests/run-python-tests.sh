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

KUNGFU_RUN=${ROOT}/bin/kungfu-run

ensure_kungfu_run() {
    if [ ! -f ${KUNGFU_RUN} ]; then
        GOBIN=$PWD/bin go install -v ./srcs/go/cmd/kungfu-run
    fi
}

ensure_kungfu_run

run_python_api_tests() {
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

measure run_python_api_tests ${ROOT}/tests/python/test_python_apis.py
