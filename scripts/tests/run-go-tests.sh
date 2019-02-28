#!/bin/sh
set -e

cd $(dirname $0)/../..

CMAKE_SOURCE_DIR=$(pwd)
export CGO_CFLAGS="-I${CMAKE_SOURCE_DIR}/srcs/cpp/include"
export CGO_LDFLAGS="-L${CMAKE_SOURCE_DIR}/lib -lkungfu-base -lstdc++"

reset_go_mod() {
    echo 'module github.com/lsds/KungFu' >go.mod
    if [ -f go.sum ]; then
        rm go.sum
    fi
}

rebuild() {
    ./configure && make
}

run_unit_tests() {
    # go test -v ./srcs/go/... # FIXME: use this

    go test -v ./srcs/go/rchannel
    go test -v ./srcs/go/plan
    go test -v ./srcs/go/monitor
    go test -v ./srcs/go/ordergroup
}

run_integration_tests() {
    export GOBIN=$(pwd)/bin
    go install -v ./tests/go/...

    local period=10ms
    env \
        KUNGFU_CONFIG_ENABLE_MONITORING=true \
        KUNGFU_CONFIG_MONITORING_PERIOD=$period \
        ./bin/test-monitor -p $period -d 1s
}

reset_go_mod
rebuild
run_unit_tests
run_integration_tests
