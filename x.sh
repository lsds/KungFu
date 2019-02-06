#!/bin/sh

set -e
cd $(dirname $0)

export RUNNER=$USER
export SRC_DIR=$(pwd)
export EXPERIMENT_SCRIPT=./examples/mnist_mlp.py

./scripts/azure/relay-machine/run-experiments.sh init-remote
./scripts/azure/relay-machine/run-experiments.sh prepare


# SLP experiments AKO
export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.slp --ako-partitions 1 --staleness 5 --kickin-time 200"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.slp --ako-partitions 1 --staleness 50 --kickin-time 200"
./scripts/azure/relay-machine/run-experiments.sh run


export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.slp --ako-partitions 1 --staleness 100 --kickin-time 200"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.slp --ako-partitions 1 --staleness 1000 --kickin-time 200"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.slp --ako-partitions 1 --staleness 5 --kickin-time 10"
./scripts/azure/relay-machine/run-experiments.sh run


export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.slp --ako-partitions 1 --staleness 5 --kickin-time 500"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.slp --ako-partitions 1 --staleness 5 --kickin-time 1000"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.slp --ako-partitions 1 --staleness 5 --kickin-time 1500"
./scripts/azure/relay-machine/run-experiments.sh run


# SLP experiments PLAIN
export EXPERIMENT_ARGS="--kungfu-strategy plain --model-name mnist.slp"
./scripts/azure/relay-machine/run-experiments.sh run


# MLP experiments AKO
export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 1 --staleness 5 --kickin-time 100"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 5 --staleness 5 --kickin-time 100"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 10 --staleness 5 --kickin-time 100"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 20 --staleness 5 --kickin-time 100"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 2 --staleness 5 --kickin-time 10"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 5 --staleness 5 --kickin-time 10"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 10 --staleness 5 --kickin-time 10"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 50 --staleness 5 --kickin-time 10"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 100 --staleness 5 --kickin-time 10"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 5 --staleness 5 --kickin-time 50"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 5 --staleness 5 --kickin-time 100"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 5 --staleness 5 --kickin-time 500"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 5 --staleness 5 --kickin-time 1000"
./scripts/azure/relay-machine/run-experiments.sh run

export EXPERIMENT_ARGS="--kungfu-strategy ako --model-name mnist.mlp --ako-partitions 5 --staleness 5 --kickin-time 1500"
./scripts/azure/relay-machine/run-experiments.sh run

# MLP experiments PLAIN
export EXPERIMENT_ARGS="--kungfu-strategy plain --model-name mnist.mlp"
./scripts/azure/relay-machine/run-experiments.sh run