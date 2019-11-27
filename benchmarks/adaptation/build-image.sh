#!/bin/sh
set -e

cd $(dirname $0)
W=$PWD

cd ../..
TAG=kungfu-adaptation-benchmark:snapshot

if [ "$1" = "run" ]; then
    docker run --rm -it $TAG
else
    docker build --rm -t $TAG -f $W/Dockerfile .
fi
