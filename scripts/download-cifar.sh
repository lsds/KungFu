#!/bin/sh

# https://www.cs.toronto.edu/~kriz/cifar.html

DATA_DIR=$HOME/var/data/cifar

download_cifar() {
    local PREFIX=https://www.cs.toronto.edu/~kriz
    local FILENAME=$1
    if [ ! -f "$1" ]; then
        curl -sLOJ ${PREFIX}/${FILENAME}
    fi
    tar -xvf ${FILENAME}
}

mkdir -p $DATA_DIR && cd $DATA_DIR

# download_cifar cifar-10-binary.tar.gz
# download_cifar cifar-100-binary.tar.gz
download_cifar cifar-10-python.tar.gz
download_cifar cifar-100-python.tar.gz
