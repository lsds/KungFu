#!/bin/sh
set -e
set -x

cd $(dirname $0)/..

build() {
    GOBIN=$(pwd)/bin go install -v ./srcs/go/kungfu-run
    ./configure && make
}

build

clean_sock() {
    for sock in $(find /tmp/ | grep kungfu-run); do
        rm -v $sock
    done
}

# clean_sock
./bin/fake-task

QUIET=-v=false

run_with_algo() {
    local ALGO=$1
    echo "running test with algorithm $ALGO"
    # clean_sock
    KUNGFU_ALLREDUCE_ALGO=$ALGO \
        ./bin/kungfu-run \
        -np=4 \
        -timeout=5s \
        ${QUIET} \
        ./bin/fake-task
}

all_algos="SIMPLE RING CLIQUE TREE"
for algo in $all_algos; do
    run_with_algo $algo
done
