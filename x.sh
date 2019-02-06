#!/bin/sh
set -e
cd $(dirname $0)
export RUNNER=$USER
export SRC_DIR=$(pwd)
export EXPERIMENT_SCRIPT=./examples/mnist_mlp.py
export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp"

./scripts/azure/relay-machine/run-experiments.sh init-remote
./scripts/azure/relay-machine/run-experiments.sh prepare
./scripts/azure/relay-machine/run-experiments.sh run
