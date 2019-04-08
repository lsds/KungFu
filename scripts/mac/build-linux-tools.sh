#!/bin/sh

set -e

cd $(dirname $0)/../..

TAG=kungfu-tools:snapshot

docker build --rm -t $TAG -f docker/Dockerfile.kungfu-tools .
id=$(docker create -t $TAG)
mkdir -p release/linux
docker cp $id:/src/kungfu/bin ./release/linux
docker rm $id
