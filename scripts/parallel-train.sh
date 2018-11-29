#!/bin/sh
set -e

cd $(dirname $0)/..

. ./scripts/install.sh

prun() {
    local n=$1
    ./bin/kungfu-run -np $n \
        python3 \
        ./examples/kungfu-train.py --use-async-sgd=1
}

echo "running ..."
prun 2

echo "done $0"
