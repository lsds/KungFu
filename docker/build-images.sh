#!/bin/sh
set -e

cd $(dirname $0)

SOURCES_LIST=sources.list.aliyun
PY_MIRROR='-i https://pypi.tuna.tsinghua.edu.cn/simple'

pack_kungfu() {
    cd ..
    tar -cvf - srcs cmake CMakeLists.txt setup.py go.mod | gzip -c >docker/kungfu.tar.gz
    cd -
}

build_image() {
    local tag=$1
    local dockerfile=$2
    local context=$3
    docker build --rm \
        --build-arg SOURCES_LIST="${SOURCES_LIST}" \
        --build-arg PY_MIRROR="${PY_MIRROR}" \
        -t ${tag} -f $dockerfile $context

}

run_example() {
    docker run --rm \
        -v $(pwd)/../examples:/examples \
        -v $HOME/var/data:/root/var/data \
        -it registry.gitlab.com/lsds-kungfu/image/kungfu:tf-cpu-ubuntu18 $@
}

# build_image registry.gitlab.com/lsds-kungfu/image/builder:ubuntu18 Dockerfile.builder-ubuntu18 .
pack_kungfu
build_image registry.gitlab.com/lsds-kungfu/image/kungfu:tf-cpu-ubuntu18 Dockerfile.tf-cpu-ubuntu18 .
run_example python3 ./examples/mnist_mlp.py
