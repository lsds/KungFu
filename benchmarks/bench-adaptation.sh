#!/bin/sh
set -e

cd $(dirname $0)

config_server_port=38080
timeout=2m

cap=16
H=127.0.0.1:$cap

kungfu_run() {
    local init_np=$1
    shift
    ../bin/kungfu-prun \
        -H ${H} \
        -np $init_np \
        -timeout ${timeout} \
        -config-server-port ${config_server_port} \
        -w \
        $@
}

if [ $(uname -s) = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
fi

TF_CPP_MIN_LOG_LEVEL=2

kungfu_run 2 python3 adaptive_trainer.py
