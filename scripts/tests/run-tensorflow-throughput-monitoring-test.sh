#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)
cd ../..
ROOT=$(pwd)

kungfu_run_flags() {
    echo -q
    echo -np 4
    echo -logdir logs
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

kungfu_run \
    env \
    KUNGFU_CONFIG_ENABLE_MONITORING=true \
    python3 \
    tests/python/integration/test_tensorflow_throughput_monitoring.py \
    --max-step 1000 \
    --data-size $((1024 * 1024))
