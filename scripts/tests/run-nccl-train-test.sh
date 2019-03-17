#!/bin/sh

set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)
cd ../..
ROOT=$(pwd)

export NCCL_HOME=$HOME/local/nccl

env \
    KUNGFU_USE_NCCL=1 \
    pip3 install --no-index -U .

KUNGFU_PRUN=${ROOT}/bin/kungfu-prun
if [ ! -f ${KUNGFU_PRUN} ]; then
    ${ROOT}/scripts/go-install.sh
fi

# FIXME: don't depend on LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib

run_nccl_experiment() {
    local np=$1
    shift
    local timeout=45s
    env \
        TF_CPP_MIN_LOG_LEVEL=1 \
        ${KUNGFU_PRUN} \
        -timeout $timeout \
        -np $np \
        python3 $@
}

run_nccl_experiment_all() {
    for np in $(seq 4); do
        run_nccl_experiment $np $@
    done
}

run_nccl_experiment_all ./tests/python/fake_tf_trainer.py
run_nccl_experiment_all ./experiments/kungfu/kf_tensorflow_synthetic_benchmark.py
