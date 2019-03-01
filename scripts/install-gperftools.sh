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

export PREFIX=$HOME/local/gperftools

mkdir -p $HOME/tmp && cd $HOME/tmp

version=2.7
filename=gperftools-${version}.tar.gz
folder=${filename%.tar.gz}
URL=https://github.com/gperftools/gperftools/releases/download/gperftools-${version}/$filename
sha1=89e3e1df674bc4ba1a9e97246b58a26a4e92d0a3

ensure_file $sha1 $filename $URL
tar -xf $filename
cd $folder

./configure --prefix=$PREFIX
make -j${NPROC} && make install
