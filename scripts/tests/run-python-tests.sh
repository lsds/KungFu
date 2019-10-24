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

measure run_tests 4 ${PYTHON} ${ROOT}/tests/python/test_python_apis.py
