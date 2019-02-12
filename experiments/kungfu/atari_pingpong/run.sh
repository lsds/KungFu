#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$(cd ../../.. && pwd)

KUNGFU_PRUN=${ROOT}/bin/kungfu-prun
if [ ! -f ${KUNGFU_PRUN} ]; then
    ${ROOT}/scripts/go-install.sh
fi

if [ $(uname -s) = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
fi

python3 ./pingpong.py --init=1

np=8
batch_size=1
timeout=1m

${KUNGFU_PRUN} \
    -timeout $timeout \
    -np $np \
    python3 \
    ./pingpong.py \
    --batch-size $batch_size
