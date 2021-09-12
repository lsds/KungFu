#!/bin/sh
set -e

cd $(dirname $0)/..
pwd
mkdir -p .build

cd .build

cmake_flags() {
    true
}

cmake ../deps/ms-elastic $(cmake_flags)
make -j $(nproc)
