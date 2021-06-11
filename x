#!/bin/sh
set -e

cd $(dirname $0)
. ./scripts/utils/measure.sh

reinstall() {
    pip3 install -U .
}

# measure reinstall
# ./experimental/rl-example/run.sh
./experimental/queue/run.sh
