#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)
cd ../..
ROOT=$(pwd)

kungfu_run_flags() {
    local config_port=9100
    echo -q
    echo -w
    echo -config-server http://127.0.0.1:$config_port/config
    echo -builtin-config-port $config_port
    echo -np 1
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

# Test CPU
kungfu_run python3 tests/python/integration/test_tensorflow_resize.py

# Test NCCL
kungfu_run python3 tests/python/integration/test_tensorflow_resize.py --use-nccl
