#!/bin/sh
set -e

if [ -z ${TAG} ]; then
    TAG=kungfu-ci-base:snapshot
fi

cd $(dirname $0)/../..

fake_trainer_flags() {
    echo kungfu-fake-adaptive-trainer
    echo --max-step 32
}

./benchmarks/adaptation/gen-compose.py --image ${TAG} --cmd "$(fake_trainer_flags)" --node-cap 4 --nodes 4 --np 1 --ttl 1m
docker-compose up
