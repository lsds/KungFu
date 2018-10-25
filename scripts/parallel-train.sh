#!/bin/sh
set -e

cd $(dirname $0)/..

pip3 install --no-index --user -U .

W1=127.0.0.1:3333
W2=127.0.0.1:3334

export KUNGFU_PEERS="$W1,$W2"

run_kungfu_task() {
    KUNGFU_TASK=$1 \
        ./examples/kungfu-train.py \
        >$1.stdout.log \
        2>$1.stderr.log
}

run_kungfu_task $W1 &
run_kungfu_task $W2 &

wait
echo "$0 done"
