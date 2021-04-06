#!/bin/sh
set -e

cd $(dirname $0)

. ../../scripts/launcher.sh

train_flags() {
    echo --batch-size 100
    echo --epochs 3
}

main() {
    rm -fr checkpoints
    python3 ./torch_mnist_example.py --epochs 0 --download # Download dataset
    erun 1 python3 torch_mnist_example.py $(train_flags)
}

main
