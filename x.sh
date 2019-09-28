#!/bin/sh
set -e

# GOROOT=$HOME/local/go
# export PATH=$GOROOT/bin:$PATH

go version

run_fake_go_trainer() {
    local np=$1
    local model=$2
    local H=127.0.0.1:$np
    echo $model
    env \
        KUNGFU_TEST_CLUSTER_SIZE=$np \
        ./bin/kungfu-prun \
        -np=$np \
        -H $H \
        -timeout=4m \
        ./bin/fake-go-trainer \
        -model $model 2>&1 | grep Img/sec
}

run_allreduce_bench() {
    local np=$1
    local model=$2
    local mode=$3
    local H=127.0.0.1:$np
    echo $model
    env \
        KUNGFU_TEST_CLUSTER_SIZE=$np \
        ./bin/kungfu-prun \
        -np=$np \
        -H $H \
        -timeout=4m \
        ./bin/bench-allreduce \
        -model $model \
        -mode $mode \
        2>&1 | grep Result
}

setup() {
    local KUNGFU_SRC=$1
    local commit=$2

    cd $KUNGFU_SRC
    git checkout master
    git pull
    git checkout -f $commit
    git clean -fdx
}

rebuild() {
    ./configure --build-tools --build-tests --with-mpi
    make -j 8

    CMAKE_SOURCE_DIR=$(pwd)
    export CGO_CFLAGS="-I${CMAKE_SOURCE_DIR}/srcs/cpp/include"
    export CGO_LDFLAGS="-L${CMAKE_SOURCE_DIR}/lib -lkungfu-base -lstdc++"
    export CGO_CXXFLAGS="-std=c++11"

    env \
        GOBIN=$(pwd)/bin \
        go install -v ./tests/go/...
}

regression() {
    # local KUNGFU_SRC=$1
    # local commit=$2

    # echo $KUNGFU_SRC $commit

    # setup $KUNGFU_SRC $commit 2>err.log >out.log
    git log --pretty=oneline -1

    rebuild 2>err.log >out.log

    # run_fake_go_trainer 4 resnet50-imagenet
    # run_fake_go_trainer 4 vgg16-imagenet

    run_allreduce_bench 4 resnet50-imagenet seq
    # run_allreduce_bench 4 vgg16-imagenet seq

    run_allreduce_bench 4 resnet50-imagenet par
    # run_allreduce_bench 4 vgg16-imagenet par
}

# KUNGFU_SRC=$HOME/Desktop/kf-old
# KUNGFU_SRC=$HOME/Desktop/kf-mh

# if [ ! -d $KUNGFU_SRC ]; then
#     git clone git@github.com:lsds/KungFu.git $KUNGFU_SRC
# fi

regression

# regression $KUNGFU_SRC c86f701cb096946b3d29e967094c324abc81d2b2 # Add benchmark for allreduce (#116)
# regression $KUNGFU_SRC 6e1e8975003e47b162eb21037fdbf4358154685e # apply chunk optimization (#117)
# regression $KUNGFU_SRC e38f3071126847130c125a3d26c19e3a9e5bbfc3 # RecvInto

echo OK >~/done