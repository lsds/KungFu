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

SCRIPT=${ROOT}/experiments/kungfu/kf_tensorflow_synthetic_benchmark.py

# FIXME: don't depend on LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib

np=4
timeout=45s

${KUNGFU_PRUN} \
    -timeout $timeout \
    -np $np \
    python3 \
    ${SCRIPT}
