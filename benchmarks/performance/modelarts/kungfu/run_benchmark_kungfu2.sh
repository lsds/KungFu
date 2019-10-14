#!/usr/bin/env bash
# https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0087.html

HOST_GPU_SLOT=8
NUM_WORKERS=16

echo $DLS_TASK_INDEX
echo $DLS_TASK_NUMBER

HOSTS=""
for ((i = 0; i < DLS_TASK_NUMBER; i++)); do
    HOST_STR="BATCH_CUSTOM${i}_HOSTS"
    HOST_VAR="${!HOST_STR}"
    IFS=':' read -r -a array <<<"$HOST_VAR"
    HOST_NAME="${array[0]}"

    if [ "$HOSTS" == "" ]; then
        HOSTS="${HOST_NAME}:${HOST_GPU_SLOT}"
    else
        HOSTS="${HOSTS},${HOST_NAME}:${HOST_GPU_SLOT}"
    fi
done

echo $HOSTS

run_experiment() {
    local np=$1
    shift

    # Each node has 8 GPUs and the NIC name is ib0
    KUNGFU_CONFIG_SHOW_DEBUG_LOG=true /KungFu/bin/kungfu-run \
        -np ${np} -H $HOSTS -nic bond0 \
        -timeout 10000s \
        $@
}

SCRIPT_PATH=/home/work/user-job-dir/src/benchmark_kungfu.py

run_experiment $NUM_WORKERS python3 $SCRIPT_PATH \
    --batch-size 64 \
    --model=ResNet50 \
    --num-iters=50
