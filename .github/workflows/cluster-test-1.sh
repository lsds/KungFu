#!/bin/sh
set -e

if [ -z ${TAG} ]; then
    TAG=kungfu-ci-base:snapshot
fi

cd $(dirname $0)/../..

./benchmarks/adaptation/gen-compose.py --image ${TAG} --ttl 20s
docker-compose up
