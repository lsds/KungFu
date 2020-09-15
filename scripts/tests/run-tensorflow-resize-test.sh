#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)
cd ../..
ROOT=$(pwd)

kungfu_run_flags() {
    echo -q
    echo -w
    echo -np 1
    echo -config-server http://127.0.0.1:9100/config
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

GOBIN=$PWD/bin \
    go install -v ./srcs/go/cmd/kungfu-config-server

kungfu_run python3 tests/python/integration/test_tensorflow_resize.py
kungfu_run python3 tests/python/integration/test_tensorflow_resize.py --use-nccl
