#!/bin/sh
set -e

cd $(dirname $0)/..
pwd

find_python() {
    if [ $(basename $(realpath $PWD/../..)) = "mindspore" ]; then
        which python3.7
    else
        which python3
    fi
}

PYTHON=$(find_python)

reload=1

elastic_run_n_flags() {
    local np=$1

    echo -q
    echo -logdir logs

    echo -w
    if [ "$reload" -eq 1 ]; then
        echo -elastic-mode reload
    fi
    echo -builtin-config-port 9100
    echo -config-server http://127.0.0.1:9100/config

    echo -np $np
}

elastic_run_n() {
    local np=$1
    shift

    $PWD/bin/kungfu-elastic-run $(elastic_run_n_flags $np) $@
}

app_flags() {
    echo --index-file $PWD/../../tf-index-1.idx.txt
    echo --max-progress 88641

    echo --global-batch-size 24
    # echo --global-batch-size $((1 << 12))

    echo --run

    if [ "$reload" -eq 1 ]; then
        echo --reload
    fi
}

main() {
    elastic_run_n 1 $PYTHON deps/ms-elastic/examples/example-elastic-worker.py $(app_flags)
}

main
