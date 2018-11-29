#!/bin/sh
set -e

cd $(dirname $0)/..

. ./scripts/install.sh

prun() {
    local np=$1
    local model_name=$2
    ./bin/kungfu-run -np $np \
        python3 \
        ./examples/kungfu-train.py --use-async-sgd=1 --model-name=$model_name
}

echo "running ..."
prun 2 mnist.slp

echo "done $0"
