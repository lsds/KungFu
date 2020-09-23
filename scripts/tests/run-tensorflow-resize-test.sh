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

start_config_server() {
    ./bin/kungfu-config-server &
}

stop_config_server() {
    curl http://127.0.0.1:9100/stop
}

GOBIN=$PWD/bin \
    go install -v ./srcs/go/cmd/kungfu-config-server

# Test CPU
start_config_server
kungfu_run python3 tests/python/integration/test_tensorflow_resize.py
stop_config_server

# Test NCCL
start_config_server
kungfu_run python3 tests/python/integration/test_tensorflow_resize.py --use-nccl
stop_config_server
