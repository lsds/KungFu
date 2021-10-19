#!/bin/sh
set -e

cd $(dirname $0)

echo "Building MLFS: $PWD"

cmake .
make -j $(nproc)
