#!/bin/sh
set -e

cd $(dirname $0)
KUNGFU_ROOT=$(pwd)/../..

timeout=2m

cap=16
H=127.0.0.1:$cap

kungfu_run() {
    local init_np=$1
    shift
    ${KUNGFU_ROOT}/bin/kungfu-run \
        -H ${H} \
        -np $init_np \
        -timeout ${timeout} \
        -w \
        $@
}

if [ $(uname -s) = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
fi

TF_CPP_MIN_LOG_LEVEL=2

kungfu_run 2 python3 adaptive_trainer.py
kungfu_run 2 python3 adaptive_trainer.py --schedule '1:16,1:1,1:16,1:1'
