#!/bin/sh
set -e

PYTHON=$(which python3)
echo "Using $PYTHON"

runner_flags() {
    echo -logdir logs
    echo -q

    echo -w
    echo -elastic-mode reload
    echo -builtin-config-port 9100
    echo -config-server http://127.0.0.1:9100/config
}

elastic_run_n() {
    local init_np=$1
    shift
    $PYTHON -m kungfu.cmd $(runner_flags) -np $init_np $@
}

app() {
    echo python3 tests/python/integration/test_elastic_reload.py
    echo --max-step 100
}

elastic_run_n 1 $(app)
