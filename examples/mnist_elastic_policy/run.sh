#!/bin/sh
set -e

cd $(dirname $0)

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -w
    echo -builtin-config-port 9100
    echo -config-server http://127.0.0.1:9100/config
    echo -np 1
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

train_flags() {
    echo --data-dir $HOME/var/data/mnist
    echo --model-dir checkpoints
    echo --num-epochs 1
}

main() {
    rm -fr checkpoints
    kungfu_run python3 main.py $(train_flags)
}

main
