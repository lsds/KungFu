#!/bin/bash
set -e

cd $(dirname $0)/../..
. ./scripts/utils/measure.sh

export PYTHONWARNINGS='ignore'
export KUNGFU_CONFIG_SHARD_HASH_METHOD=NAME
export KUNGFU_CONFIG_LOG_LEVEL=ERROR
export TF_CPP_MIN_LOG_LEVEL=2

kungfu_flags() {
    local H=127.0.0.1:8
    local init_np=1

    echo -q
    echo -H $H
    echo -np $init_np
    echo -logdir logs
    echo -w
    echo -config-server http://127.0.0.1:9100/get
}

flags() {
    local schedule_flags=$(./tests/python/integration/gen_schedule.py)
    echo $schedule_flags
    echo --data-dir $HOME/var/data/mnist
}

main() {
    rm -fr checkpoints
    env \
        PATH=$PWD/bin:$PATH \
        kungfu-run $(kungfu_flags) python3 tests/python/integration/test_elastic_estimator.py $(flags)
}

measure main
