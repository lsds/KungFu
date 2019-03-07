#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)

. ./utils/ensure_file.sh

mkdir -p $HOME/tmp
cd $HOME/tmp

version=2.4.2-1
filename=nccl-${version}.tar.gz
URL=https://github.com/NVIDIA/nccl/archive/v${version}.tar.gz
folder=${filename%.tar.gz}

ensure_file 0dcac5994c54af839e2e1646e88d12c2fe77b338 $filename $URL
tar -xf $filename
cd $folder

if [ -z "$PREFIX" ]; then
    PREFIX=$HOME/local/nccl
fi
export PREFIX

make -j$(nproc)
make install
