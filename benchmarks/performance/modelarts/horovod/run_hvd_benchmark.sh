#!/usr/bin/env bash
# This script runs the Horovod benchmark for ResNet-50 on the modelarts platform.
# 
# This script assumes run on 2 DGX-1 machine which has 8 GPUs. In total, we have 16 GPUs. Check the gen_hostfile() function.
# We assume that the user has copy a folder named benchmarks. This folder needs to be placed in the Huawei OBS service.
# The path to the benchmarks folder is: /home/work/user-job-dir/benchmarks/
# The benchmarks folder has the followin files:
# - horovod/kube-plm-rsh-agent
# - horovod/hvd_tensorflow_synthetic_benchmark.py
set -x

cd /home/work/user-job-dir/

# Modify script path to point to benchmark script
SCRIPT_PATH=KungFu/performance/hvd_tensorflow_synthetic_benchmark.py
# Modify RSH agent path to point to kube-plm-rsh-agent file
RSH_AGENT_PATH=KungFu/performance/modelarts/horovod/kube-plm-rsh-agent
# Modify hostfile indicating where pod characteristics are located
HOST_FILE_PATH=KungFu/performance/modelarts/horovod/hostfile

# KungFu/performance/modelarts/horovod/kube-plm-rsh-agent
echo $PWD
ls $PWD
ls $PWD/KungFu
ls $PWD/KungFu/performance
ls $PWD/KungFu/performance/modelarts
ls $PWD/KungFu/performance/horovod


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
        echo "$pod slots=8"
    done
}

gen_hostfile >$HOST_FILE_PATH

MPI_HOME=$HOME/local/openmpi

run_experiment(){
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

export TF_CPP_MIN_LOG_LEVEL=1

if [ "$DLS_TASK_INDEX" = "0" ]
then
    batch_size=256
    nps="16"
    run_experiment $nps python3 $script --batch-size=$batch_size --model=ResNet50 --num-iters=50
else
    sleep 5d
fi