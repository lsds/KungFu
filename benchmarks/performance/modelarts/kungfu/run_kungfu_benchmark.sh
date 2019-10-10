#!/bin/sh
# This script runs the keras benchmark on 2 ModelArts nodes
# Run this script on every node in the cluster.
set -e

run_experiment(){
local np=$1
shift

# Assuming the IP address for these 2 nodes are: 169.254.128.207, 169.254.128.185
# Each node has 8 GPUs and the NIC name is ib0
kungfu-prun \
-np ${np} -H 169.254.128.207:8,169.254.128.185:8 -nic ib0 \
-timeout 10000s \
$@
}

export TF_CPP_MIN_LOG_LEVEL=1

# Run the experiment with 16 KungFu nodes.
# The model is ResNet-50, batch size is 64, and the experiments runs for 50 iterations.

# Modify script path to point to your benchmark script
SCRIPT_PATH=$PWD/KungFu/performance/kf_tensorflow_synthetic_benchmark.py

run_experiment 16 python3 $SCRIPT_PATH \
--batch-size 64 \
--model=ResNet50 \
--num-iters=50