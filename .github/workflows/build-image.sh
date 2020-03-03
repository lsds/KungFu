#!/bin/sh
set -e

cd $(dirname $0)
W=$PWD

cd ../..

BASE_TAG=github-ci-base:latest
TAG=kungfu-ci-base:snapshot

if [ "$1" = "run" ]; then
    docker run --rm -it $TAG
else
    docker build --rm -t $BASE_TAG -f .github/Dockerfile.base .
    docker build --rm -t $TAG -f .github/Dockerfile.kungfu .
fi
