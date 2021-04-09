#!/bin/sh
set -e

cd $(dirname $0)

. ../../scripts/launcher.sh

train_flags() {
    echo --batch-size 100
    echo --epochs 1
}

main() {
    rm -fr checkpoints
    if [ ! -d data ]; then
        srun python3 ./torch_mnist_example.py --epochs 0 --download # Download dataset took 6s
    fi

    erun 1 python3 torch_mnist_example.py $(train_flags)
}

main
