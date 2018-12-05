#!/bin/sh

# http://yann.lecun.com/exdb/mnist/

download_mnist_data(){
    local prefix=http://yann.lecun.com/exdb/mnist
    [ ! -f "$1" ] && curl -sOJ $prefix/$1
}

download_mnist_data train-images-idx3-ubyte.gz
download_mnist_data train-labels-idx1-ubyte.gz
download_mnist_data t10k-images-idx3-ubyte.gz
download_mnist_data t10k-labels-idx1-ubyte.gz

gzip -dfk *.gz
