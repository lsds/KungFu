#!/bin/sh
set -e

cd $(dirname $0)/..
ROOT=$(pwd)

CMAKE_SOURCE_DIR=$(pwd)

reset_go_mod() {
    echo 'module github.com/lsds/KungFu' >go.mod
    if [ -f go.sum ]; then
        rm go.sum
    fi
}

get_go_source() {
    local URL=$1
    local DIR=$2
    if [ ! -d $DIR ]; then
        git clone $URL $DIR
    fi
}

go_install_old() {
    export GOPATH=${ROOT}/gopath

    if [ -z "${GO_X_CRYPTO_GIT_URL}" ]; then
        GO_X_CRYPTO_GIT_URL=https://github.com/golang/crypto.git
    fi

    get_go_source $GO_X_CRYPTO_GIT_URL $GOPATH/src/golang.org/x/crypto

    gomod=$(head -n 1 ${ROOT}/go.mod | awk '{print $2}')
    src_loc=$GOPATH/src/$gomod
    mkdir -p $(dirname $src_loc)
    [ -d $src_loc ] && rm $src_loc
    ln -v -s $ROOT $src_loc

    ./configure $@ && make
    env \
        GO111MODULE=off \
        GOBIN=$(pwd)/bin \
        go install -v ./srcs/go/cmd/...
}

go_clean() {
    go clean -cache ./...
}

go_install() {
    env \
        GOPROXY=https://goproxy.io \
        GOBIN=$(pwd)/bin \
        go install -v ./srcs/go/cmd/...
}

go_clean
reset_go_mod
go_install
reset_go_mod
