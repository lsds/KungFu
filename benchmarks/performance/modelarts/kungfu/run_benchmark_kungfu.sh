#!/bin/sh
# This script runs the keras benchmark on 2 ModelArts nodes
# Run this script on every node in the cluster.
set -e

SCRIPT_PATH=/home/work/user-job-dir/src/benchmark_kungfu.py
BATCH_SIZE=64
NUM_WORKERS=16
MODEL="ResNet50"
NUM_ITERS=100

# Assume each host has 8 GPUs.
WORKER_HOSTS="169.254.128.207:8,169.254.128.185:8"

run_experiment() {
    local np=$1
    shift

    # Use the IB NIC ib0. This IB NIC is running in the ib2ip mode.
    kungfu-prun \
        -np ${np} -H $WORKER_HOSTS -nic ib0 \
        $@
}

export TF_CPP_MIN_LOG_LEVEL=1

run_experiment $NUM_WORKERS python3 $SCRIPT_PATH \
    --batch-size=$BATCH_SIZE \
    --model=$MODEL \
    --num-iters=$NUM_ITERS
