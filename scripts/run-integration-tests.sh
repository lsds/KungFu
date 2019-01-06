#!/bin/sh
set -e
set -x

cd $(dirname $0)/..

CMAKE_SOURCE_DIR=$(pwd)
export CGO_CFLAGS="-I${CMAKE_SOURCE_DIR}/srcs/cpp/include"
export CGO_LDFLAGS="-L${CMAKE_SOURCE_DIR}/lib -lkungfu-base -lstdc++"

./scripts/go-install.sh

QUIET=-v=false

run_with_algo() {
    local ALGO=$1
    local np=4
    local H=127.0.0.1:$np
    echo "running test with algorithm $ALGO"
    KUNGFU_TEST_CLUSTER_SIZE=$np \
        ./bin/kungfu-prun \
        -np=$np \
        -algo=$ALGO \
        -H $H \
        -timeout=5s \
        ${QUIET} \
        ./bin/fake-task
}

all_algos="SIMPLE RING CLIQUE TREE"
for algo in $all_algos; do
    run_with_algo $algo
done
