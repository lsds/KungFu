#!/bin/sh
set -e

# http://yann.lecun.com/exdb/mnist/

if [ ! -z "${DATA_DIR}" ]; then
    mkdir -p $DATA_DIR
    cd $DATA_DIR
fi

mkdir -p mnist
cd mnist

download_mnist_data() {
    local prefix=http://yann.lecun.com/exdb/mnist
    export DATA_MIRROR_PREFIX=https://storage.googleapis.com/cvdf-datasets
    # export DATA_MIRROR_PREFIX=https://kungfudata.blob.core.windows.net/data
    if [ ! -z ${DATA_MIRROR_PREFIX} ]; then
        prefix=${DATA_MIRROR_PREFIX}/mnist
    fi
    if [ ! -f "$1" ]; then
        local url=$prefix/$1
        echo "download $1 from $url"
        curl -vOJ $url
    fi
}

download_mnist_data train-images-idx3-ubyte.gz &
download_mnist_data train-labels-idx1-ubyte.gz &
download_mnist_data t10k-images-idx3-ubyte.gz &
download_mnist_data t10k-labels-idx1-ubyte.gz &
wait

gzip -dfk *.gz
