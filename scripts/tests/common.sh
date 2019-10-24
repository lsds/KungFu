#!/bin/sh
set -e

. ./scripts/utils/measure.sh

KUNGFU_RUN=${ROOT}/bin/kungfu-run

ensure_kungfu_run() {
    if [ ! -f ${KUNGFU_RUN} ]; then
        GOBIN=$PWD/bin go install -v ./srcs/go/cmd/kungfu-run
    fi
}

run_tests() {
    local max_np=$1
    shift
    for np in $(seq $max_np); do
        local hosts=127.0.0.1:$np
        $KUNGFU_RUN \
            -np $np \
            -H $hosts \
            -q \
            $@
    done
}

ensure_kungfu_run
