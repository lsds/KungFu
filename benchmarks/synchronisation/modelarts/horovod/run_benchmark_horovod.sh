#!/usr/bin/env bash
# This script runs the Horovod benchmark for ResNet-50 on the modelarts platform.
# This script assumes run on 2 DGX-1 machine which has 8 GPUs. In total, we have 16 GPUs. Check the gen_hostfile() function.
# We assume that the user has copy the performance folder into the OBS service.
set -x

WORKER_GPU_SLOT=8
BATCH_SIZE=64
MODEL="ResNet50"
NUM_ITERS=100

# Modify script path to point to benchmark script (please set aboslute path)
SCRIPT_PATH=/home/work/user-job-dir/src/benchmark_horovod.py
# Modify RSH agent path to point to kube-plm-rsh-agent file (please set aboslute path)
RSH_AGENT_PATH=/home/work/user-job-dir/src/kube_plm_rsh_agent
# Modify hostfile indicating where pod characteristics are located (please set aboslute path)
HOST_FILE_PATH=/home/work/user-job-dir/src/hostfile

chmod +x $RSH_AGENT_PATH

KUBE_SA_CONFIG=/var/run/secrets/kubernetes.io/serviceaccount
if [ -d $KUBE_SA_CONFIG ]; then
    NAMESPACE=$(cat $KUBE_SA_CONFIG/namespace)
    TOKEN=$(cat $KUBE_SA_CONFIG/token)
fi

kubectl config set-cluster this --server https://kubernetes/ --certificate-authority=$KUBE_SA_CONFIG/ca.crt
kubectl config set-credentials me --token "$TOKEN"
kubectl config set-context me@this --cluster=this --user=me --namespace "$NAMESPACE"
kubectl config use me@this

kubectl get pods
kubectl config view

gen_hostfile() {
    pods=$(kubectl get pods -o name | grep job | awk -F '/' '{print $2}')
    for pod in $pods; do
        echo "$pod slots=$WORKER_GPU_SLOT"
    done
}

gen_hostfile >$HOST_FILE_PATH

MPI_HOME=$HOME/local/openmpi

run_distributed_experiment() {
    local np=$1
    shift

    mpirun --allow-run-as-root -np ${np} \
        -mca plm_rsh_agent $RSH_AGENT_PATH \
        --hostfile $HOST_FILE_PATH \
        --bind-to socket \
        -x LD_LIBRARY_PATH \
        -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=ib0,bond0,eth0 -x NCCL_SOCKET_FAMILY=AF_INET -x NCCL_IB_DISABLE=0 \
        -x HOROVOD_MPI_THREADS_DISABLE=1 \
        -mca pml ob1 -mca btl ^openib \
        -mca plm_rsh_no_tree_spawn true \
        -mca btl_tcp_if_include 192.168.0.0/16 \
        $@
}

run_local_experiment() {
    local np=$1
    shift

    mpirun --allow-run-as-root -np ${np} \
        -mca plm_rsh_agent $RSH_AGENT_PATH \
        --bind-to socket \
        -x LD_LIBRARY_PATH \
        -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=ib0,bond0,eth0 -x NCCL_SOCKET_FAMILY=AF_INET -x NCCL_IB_DISABLE=0 \
        -x HOROVOD_MPI_THREADS_DISABLE=1 \
        -mca pml ob1 -mca btl ^openib \
        -mca plm_rsh_no_tree_spawn true \
        -mca btl_tcp_if_include 192.168.0.0/16 \
        $@
}

export TF_CPP_MIN_LOG_LEVEL=1

if [ "$DLS_TASK_NUMBER" == "1" ]; then
    run_local_experiment $WORKER_GPU_SLOT python3 $SCRIPT_PATH --batch-size=$BATCH_SIZE --model=$MODEL --num-iters=$NUM_ITERS
else
    if [ "$DLS_TASK_INDEX" = "0" ]; then
        NUM_WORKERS=$((DLS_TASK_INDEX * WORKER_GPU_SLOT))
        run_distributed_experiment $NUM_WORKERS python3 $SCRIPT_PATH --batch-size=$BATCH_SIZE --model=$MODEL --num-iters=$NUM_ITERS
    else
        sleep 5d
    fi
fi
