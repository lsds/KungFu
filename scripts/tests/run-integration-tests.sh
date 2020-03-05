#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)
cd ../..
. ./scripts/utils/measure.sh

run_fake_cluster() {
    local np=$1
    shift
    local STRATEGY=$1
    shift

    local H=127.0.0.1:$np
    local QUIET=-v=false

    echo "running test with graph strategy $STRATEGY"
    ./bin/kungfu-run \
        -q \
        -np=$np \
        -strategy="${STRATEGY}" \
        -H $H \
        -timeout=5s \
        ${QUIET} \
        $@
}

test_all() {
    all_strategies="STAR RING CLIQUE TREE BINARY_TREE BINARY_TREE_STAR MULTI_BINARY_TREE_STAR AUTO"
    for np in $(seq 4); do
        for STRATEGY in $all_strategies; do
            run_fake_cluster $np $STRATEGY ./bin/fake-agent
            run_fake_cluster $np $STRATEGY ./bin/test-p2p-apis
        done
    done
}

measure test_all
