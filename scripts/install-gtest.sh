#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)

. ./utils/ensure_file.sh

if [ $(uname) = "Darwin" ]; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=$(nproc)
fi

mkdir -p $HOME/tmp
cd $HOME/tmp

version=1.8.1
filename=googletest-release-${version}.tar.gz
folder=${filename%.tar.gz}

URL=https://github.com/google/googletest/archive/release-${version}.tar.gz

ensure_file 152b849610d91a9dfa1401293f43230c2e0c33f8 $filename $URL

tar -xf $filename
cd $folder

PREFIX=$HOME/local/gtest

mkdir -p build
cd build
cmake .. \
    -DGFLAGS_BUILD_SHARED_LIBS=ON \
    -DINSTALL_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX:PATH=$PREFIX

make -j${NPROC} && make install
