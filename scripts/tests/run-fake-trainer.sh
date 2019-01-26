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

run_fake_trainer() {
    # local ALGO=
    local np=$1
    local H=127.0.0.1:$np
    KUNGFU_TEST_CLUSTER_SIZE=$np \
        ./bin/kungfu-prun \
        -np=$np \
        -algo="${ALGO}" \
        -H $H \
        -timeout=120s \
        ${QUIET} \
        ./bin/fake-trainer
}

run_fake_trainer_all() {
    for np in $(seq 4); do
        measure run_fake_trainer $np
    done
}

measure reinstall

export KUNGFU_CONFIG_LOG_CONFIG_VARS=true
measure run_fake_trainer_all
