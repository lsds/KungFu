#!/usr/bin/env bash

ifconfig -a

nvidia-smi

WORK_DIR=/home/work/user-job-dir/benchmark_ps

HOSTS=("169.254.128.207" "169.254.128.185")
# HOSTS=("127.0.0.1")
PS_PORT=20000
WORKER_PORT=30000
PS_SLOT=8
WORKER_SLOT=8

# Create the host list for parameter servers
PS_HOSTS=""
for host in "${HOSTS[@]}"; do
    for ((i = 0; i < PS_SLOT; i++)); do
        port=$((PS_PORT + i))
        str="${host}:${port}"
        if [ "$PS_HOSTS" == "" ]; then
            PS_HOSTS="${str}"
        else
            PS_HOSTS="${PS_HOSTS},${str}"
        fi
    done
done

echo $PS_HOSTS

WORKER_HOSTS=""
for host in "${HOSTS[@]}"; do
    for ((i = 0; i < WORKER_SLOT; i++)); do
        port=$((WORKER_PORT + i))
        str="${host}:${port}"
        if [ "$WORKER_HOSTS" == "" ]; then
            WORKER_HOSTS="${str}"
        else
            WORKER_HOSTS="${WORKER_HOSTS},${str}"
        fi
    done
done

echo $WORKER_HOSTS

NUM_HOSTS=${#HOSTS[@]}
if [ "$NUM_HOSTS" == "1" ]; then
    # DLS_TASK_INDEX starts from 1 in the 1-node ModerlArts cluster
    HOST_INDEX=$((DLS_TASK_INDEX - 1))
else
    # DLS_TASK_INDEX starts from 0 in the multi-node ModerlArts cluster
    HOST_INDEX=$DLS_TASK_INDEX
fi

WORKER_TASK_INDEX_OFFSET=$((HOST_INDEX * WORKER_SLOT))
for ((i = 0; i < WORKER_SLOT; i++)); do
    worker_task_index=$((WORKER_TASK_INDEX_OFFSET + i))
    echo $worker_task_index

    CUDA_VISIBLE_DEVICES="$i" python3 $WORK_DIR/benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=$worker_task_index &
done

PS_TASK_INDEX_OFFSET=$((HOST_INDEX * PS_SLOT))
for ((i = 0; i < PS_SLOT; i++)); do
    ps_task_index=$((PS_TASK_INDEX_OFFSET + i))
    echo $ps_task_index

    CUDA_VISIBLE_DEVICES="" python3 $WORK_DIR/benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=ps --task_index=$ps_task_index &
done

wait
