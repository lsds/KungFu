#!/bin/sh
set -e

cd $(dirname $0)/../..

reset_go_mod() {
    echo 'module github.com/lsds/KungFu' >go.mod
    if [ -f go.sum ]; then
        rm go.sum
    fi
}

./configure
make

CMAKE_SOURCE_DIR=$(pwd)
export CGO_CFLAGS="-I${CMAKE_SOURCE_DIR}/srcs/cpp/include"
export CGO_LDFLAGS="-L${CMAKE_SOURCE_DIR}/lib -lkungfu-base -lstdc++"

# reset_go_mod
GOBIN=$(pwd)/bin go install -v ./tests/go/...
# reset_go_mod

./bin/fake-controller \
    -timeout=120s \
    -np=3 \
    ./bin/fake-go-trainer \
    -model=mnist-slp \
    -step-per-iter=3 \
    -control
