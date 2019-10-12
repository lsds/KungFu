#!/bin/sh
set -e

cd $(dirname $0)/../..

reset_go_mod() {
    echo 'module github.com/lsds/KungFu' >go.mod
    if [ -f go.sum ]; then
        rm go.sum
    fi
}

rebuild() {
    env \
        GOBIN=$(pwd)/bin \
        go install -v ./...
}

run_unit_tests() {
    go test -v ./srcs/go/...
}

run_integration_tests() {
    local period=10ms
    env \
        KUNGFU_CONFIG_ENABLE_MONITORING=true \
        KUNGFU_CONFIG_MONITORING_PERIOD=$period \
        ./bin/test-monitor -p $period -d 1s

    local np=4
    local H=127.0.0.1:$np
    ./bin/kungfu-run \
        -H $H \
        -np $np \
        ./bin/test-public-apis
}

reset_go_mod
rebuild
run_unit_tests
run_integration_tests
