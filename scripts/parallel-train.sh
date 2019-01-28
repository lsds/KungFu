#!/bin/sh
set -e

cd $(dirname $0)/..

np=2
hosts=127.0.0.1:$np
ip=127.0.0.1

parse_args() {
    if [ ! -z "$1" ]; then
        np=$1
    fi

    if [ ! -z "$2" ]; then
        hosts=$2
    fi

    if [ ! -z "$3" ]; then
        ip=$3
    fi
}

prun() {
    local self=$1
    local np=$2
    local model_name=$3

    if [ $(uname -s) = "Darwin" ]; then
        export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
    fi

    echo "$self $np $model_name"
    ./bin/kungfu-prun \
        -np $np \
        -H $hosts \
        -self $self \
        -timeout 20s \
        python3 \
        ./examples/mnist_mlp.py \
        --use-kungfu=1 \
        --n-epochs 1 \
        --batch-size=500 \
        --model-name=$model_name
}

parse_args $@
echo "running ..."
prun $ip $np mnist.slp
#  >prun.stdout.log 2>prun.stderr.log
echo "done $0"
