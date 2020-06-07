#!/bin/bash

KUNGFU_BIN=$HOME/KungFu/bin
OUTPUT_DIR=./tmp

$KUNGFU_BIN/kungfu-config-server-example \
    -init ./init.json &

$KUNGFU_BIN/kungfu-run \
    -w \
    -H 127.0.0.1:4 \
    -config-server http://127.0.0.1:9100/get \
    -np 4 \
    -logfile kungfu-run.log \
    -logdir $OUTPUT_DIR \
    python3 mnist_slp_estimator.py

pkill -P $$
