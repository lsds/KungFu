#!/usr/bin/env bash
# https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0087.html

HOST_GPU_SLOT=0
NUM_WORKERS=$DLS_TASK_NUMBER

echo $DLS_TASK_INDEX
echo $DLS_TASK_NUMBER

HOSTS=""
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
