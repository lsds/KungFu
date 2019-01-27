#!/bin/sh
set -e

cd $(dirname $0)/../..

CMAKE_SOURCE_DIR=$(pwd)
export CGO_CFLAGS="-I${CMAKE_SOURCE_DIR}/srcs/cpp/include"
export CGO_LDFLAGS="-L${CMAKE_SOURCE_DIR}/lib -lkungfu-base -lstdc++"

# go test -v ./srcs/go/...
./configure && make
go test -v ./srcs/go/rchannel
go test -v ./srcs/go/plan
