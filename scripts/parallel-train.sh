#!/bin/sh
set -e

cd $(dirname $0)/..

hosts=127.0.0.1
ip=127.0.0.1
np=2

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

    echo "$self $np $model_name"
    ./bin/kungfu-run \
        -np $np \
        -hosts $hosts \
        -self $self \
        -m 4 \
        -timeout 20s \
        python3 \
        ./examples/kungfu-train.py \
        --use-async-sgd=1 \
        --n-epochs 1 \
        --batch-size=500 \
        --model-name=$model_name
}

parse_args $@
echo "running ..."
prun $ip $np mnist.slp
#  >prun.stdout.log 2>prun.stderr.log
echo "done $0"
