#!/bin/sh
set -e

cd $(dirname $0)/../..

CMAKE_SOURCE_DIR=$(pwd)

reset_go_mod() {
    echo 'module github.com/lsds/KungFu' >go.mod
    if [ -f go.sum ]; then
        rm go.sum
    fi
}

rebuild() {
    env \
        GOBIN=$(pwd)/bin \
        go install -v ./tests/go/...
}

prun() {
    local np=$1
    shift
    local H="127.0.0.1:$np"
    ./bin/kungfu-run \
        -timeout 1m \
        -H $H \
        -np $np \
        $@
}

reset_go_mod
rebuild

prun 4 ./bin/kungfu-bench-allreduce -model resnet50-imagenet -mode seq
prun 4 ./bin/kungfu-bench-allreduce -model resnet50-imagenet -mode par
prun 4 ./bin/kungfu-bench-allreduce -model vgg16-imagenet -mode seq
prun 4 ./bin/kungfu-bench-allreduce -model vgg16-imagenet -mode par
