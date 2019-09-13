#!/bin/sh
set -e

cd $(dirname $0)

config_server_port=38080

kungfu_run() {
    local init_np=$1
    shift
    ../bin/kungfu-prun \
        -np $init_np \
        -timeout 2m \
        -config-server-port ${config_server_port} \
        -w \
        $@
}

if [ $(uname -s) = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
fi

TF_CPP_MIN_LOG_LEVEL=2

kungfu_run 2 python3 adaptive_trainer.py
