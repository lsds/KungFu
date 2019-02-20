#!/bin/bash
# FIXME: use /bin/sh

set -e

cd $(dirname $0)

TAG=registry.gitlab.com/lsds-kungfu/image/builder:ubuntu18

SOURCES_LIST=sources.list.aliyun
PY_MIRROR='-i https://pypi.tuna.tsinghua.edu.cn/simple'

if [[ -f /etc/apt/sources.list ]] && [[ -n $(cat /etc/apt/sources.list | grep azure) ]]; then
    SOURCES_LIST=sources.list.azure
    PY_MIRROR=
fi

docker build --rm \
    --build-arg SOURCES_LIST="${SOURCES_LIST}" \
    --build-arg PY_MIRROR="${PY_MIRROR}" \
    -t ${TAG} -f Dockerfile.builder-ubuntu18 .

docker push ${TAG}
