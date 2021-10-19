#!/bin/sh

set -e

# Build executables and index files

./deps/build.sh
./deps/mlfs/build.sh
./build-squad-index.sh
./build-imagenet-index.sh
