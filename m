#!/bin/sh
set -e

reset_go_mod() {
    echo 'module github.com/lsds/KungFu' >go.mod
    if [ -f go.sum ]; then
        rm go.sum
    fi
}

export CMAKE_SOURCE_DIR=$(pwd)
export GOBIN=$(PWD)/bin

go_build() {
    go clean -cache ./...
    # go install -v ./srcs/go/cmd/...

    # export CGO_CFLAGS="-DENABLE_F16 -mf16c -mavx"
    # export CGO_CFLAGS="-mf16c -mavx -DENABLE_F16"
    # export CGO_CFLAGS="-DENABLE_F16"
    # go test -v ./srcs/go/kungfubase/...
    # go build -v -buildmode=c-archive ${CMAKE_SOURCE_DIR}/srcs/go/libkungfu-comm
}

reset_go_mod
./configure --build-tools --build-tensorflow-ops --build-tests
go install -v ./...
make -j$(nproc)

reset_go_mod
# go_build
reset_go_mod
