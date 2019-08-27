#!/bin/sh

set -e
cd $(dirname $0)/..

localhost=127.0.0.1
H=${localhost}:8

timeout=600s
logfile=prun.log

config_server_port=38080
watch_period=3s

prun() {
    local np=$1
    shift
    ./bin/kungfu-prun \
        -H "${H}" \
        -np "${np}" \
        -timeout "${timeout}" \
        -w -k=0 \
        -watch-period "${watch_period}" \
        -log-file "${logfile}" \
        -config-server-port ${config_server_port} \
        $@
}

if [ $(uname -s) = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
fi

init_np=4

prun $init_np python3 \
    ./examples/dynamic_train_mnist.py \
    --batch-size 5000
