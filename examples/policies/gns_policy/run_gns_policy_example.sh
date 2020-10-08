#!/bin/sh
set -e

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 2
    echo -w
    echo -config-server http://127.0.0.1:9100/config
    echo -builtin-config-port 9100
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

train_flags() {
    echo --model ResNet50
    echo --epoch-size 1024
    echo --epoch-num 3
    echo --max-train-steps 100
    echo --init-batch-size 32
    echo --model-dir checkpoints
    # echo --fake-data
}

main() {
    kungfu_run python3 examples/policies/gns_policy/train.py $(train_flags)
}

main

# python3 examples/policies/gns_policy/test_input_data.py
