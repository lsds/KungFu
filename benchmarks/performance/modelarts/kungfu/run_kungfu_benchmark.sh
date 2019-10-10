#!/bin/sh
# This script runs the keras benchmark on 2 ModelArts nodes
#
# Assuming the IP address for these 2 nodes are: 169.254.128.207, 169.254.128.185
# Each node has 8 GPUs and the NIC name is ib0
#
# Run this script on every node in the cluster.
set -e

run_experiment(){
shift
kungfu-prun \
-np 16 -H 169.254.128.207:8,169.254.128.185:8 -nic ib0 \
-timeout 10000s \
$@
}

export TF_CPP_MIN_LOG_LEVEL=1

# ResNet-50
run_experiment python3 kf_tensorflow_synthetic_benchmark.py \
--batch-size 64 \
--model=ResNet50 \
--kungfu=model-ave \
--kungfu-fuse-variables=True \
--num-iters=50