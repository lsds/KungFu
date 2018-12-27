#!/bin/sh
set -e
set -x

cd $(dirname $0)/..

GOBIN=$(pwd)/bin go install -v ./srcs/go/kungfu-run
./configure && make

./bin/fake-task

run_with_algo() {
    local ALGO=$1
    echo "running test with algorithm $ALGO"
    KUNGFU_ALLREDUCE_ALGO=$ALGO \
        ./bin/kungfu-run \
        -np=4 \
        -timeout=5s \
        -v=false \
        ./bin/fake-task
}

all_algos="SIMPLE RING CLIQUE TREE"
for algo in $all_algos; do
    run_with_algo $algo
done
