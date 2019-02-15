#!/bin/sh

set -e
cd $(dirname $0)

export RUNNER=$USER
export SRC_DIR=$(pwd)
export EXPERIMENT_SCRIPT=./examples/logistic_regression_mnist.py

./scripts/azure/relay-machine/run-experiments.sh init-remote
./scripts/azure/relay-machine/run-experiments.sh prepare

# try with 4 layers after
# for i in {2..8..2} 
# do
    
# done

# MLP experiments AKO
export EXPERIMENT_ARGS="--kungfu-strategy ako --num-layers 2 --ako-partitions 1 --staleness 1 --kickin-time 0"
./scripts/azure/relay-machine/run-experiments.sh run

# export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --num-layers 2 --ako-partitions 4 --staleness 1 --kickin-time 0"
# ./scripts/azure/relay-machine/run-experiments.sh run

# export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --num-layers 2 --ako-partitions 8 --staleness 1 --kickin-time 0"
# ./scripts/azure/relay-machine/run-experiments.sh run


# # MLP experiments synch SGD
# export EXPERIMENT_ARGS="--kungfu-strategy plain --model-name mnist.mlp"
# ./scripts/azure/relay-machine/run-experiments.sh run