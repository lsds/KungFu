#!/bin/sh
set -e

cd $(dirname $0)/../../
TAG=kungfu-adaptation-benchmark:snapshot

if [ "$1" = "run" ]; then
    docker run --rm -it $TAG
else
    docker build --rm -t $TAG -f ./benchmarks/adaptive/Dockerfile .
fi
