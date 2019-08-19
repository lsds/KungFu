#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)
cd ../..
. ./scripts/utils/measure.sh

reinstall() {
    CMAKE_SOURCE_DIR=$(pwd)
    export CGO_CFLAGS="-I${CMAKE_SOURCE_DIR}/srcs/cpp/include"
    export CGO_LDFLAGS="-L${CMAKE_SOURCE_DIR}/lib -lkungfu-base -lstdc++"

    ./scripts/go-install.sh
}

run_fake_cluster() {
    local np=$1
    shift
    local ALGO=$1
    shift

    local H=127.0.0.1:$np
    local QUIET=-v=false

    echo "running test with algorithm $ALGO"
    KUNGFU_TEST_CLUSTER_SIZE=$np \
        ./bin/kungfu-prun \
        -np=$np \
        -algo="${ALGO}" \
        -H $H \
        -timeout=5s \
        ${QUIET} \
        $@
}

test_all() {
    all_algos="STAR RING CLIQUE TREE"
    for np in $(seq 4); do
        for algo in $all_algos; do
            run_fake_cluster $np $algo ./bin/fake-agent
            run_fake_cluster $np $algo ./bin/test-p2p-apis
        done
    done
}

measure reinstall
measure test_all
