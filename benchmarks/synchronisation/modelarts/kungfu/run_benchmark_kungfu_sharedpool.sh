#!/bin/sh
set -e

get_port() {
    local HOST_STR="BATCH_CUSTOM${DLS_TASK_INDEX}_HOSTS"
    local HOST_VAR="${!HOST_STR}"
    IFS=':' read -r -a array <<<"$HOST_VAR"
    # HOST_NAME="${array[0]}"
    PORT="${array[1]}"
    echo $PORT
}

# export KUNGFU_CONFIG_SHOW_DEBUG_LOG=true

gen_peers() {
    local HOST_GPU_SLOT=0
    local HOSTS=""
    for ((i = 0; i < DLS_TASK_NUMBER; i++)); do
        HOST_STR="BATCH_CUSTOM${i}_HOSTS"
        HOST_VAR="${!HOST_STR}"
        # IFS=':' read -r -a array <<<"$HOST_VAR"
        # HOST_NAME="${array[0]}"

        if [ "$HOSTS" == "" ]; then
            HOSTS="${HOST_VAR}:${HOST_GPU_SLOT}"
        else
            HOSTS="${HOSTS},${HOST_VAR}:${HOST_GPU_SLOT}"
        fi
    done
    echo $HOSTS
}

KUNGFU_RUN=/KungFu/bin/kungfu-run
SCRIPT=/home/work/user-job-dir/src/benchmark_kungfu.py

run() {
    local P=$(gen_peers)
    local port=$(get_port)
    local nic=eth0
    echo "Using port=${port}"
    $KUNGFU_RUN -H "" -P $P -nic $nic -port ${port} $@
}

# run python3 -m kungfu.examples
run_all() {
    MODELS="ResNet50 VGG16"
    KUNGFU="sync-sgd async-sgd"
    for model in $(echo $MODELS); do
        for kungfu in $(echo $KUNGFU); do
            echo "Using model=$model, kungfu=$kungfu"
            run python3 $SCRIPT --kungfu=$kungfu --model=$model --batch-size=64
        done
    done
}

run_all
