#!/bin/sh

set -e

export LC_ALL=C

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)

. ./utils/ensure_file.sh

if [ $(uname) = "Darwin" ]; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=$(nproc)
fi

# https://www.python.org/downloads/
major=3
minor=6
patch=4
version=${major}.${minor}.${patch}

PYTHON_HOME=$HOME/local/python-${major}.${minor}.${patch}

TMP_DIR=$HOME/tmp
mkdir -p $TMP_DIR && cd $TMP_DIR

filename=Python-$version.tar.xz
folder=${filename%.tar.xz}
URL=https://www.python.org/ftp/python/$version/$filename
SHA1=36a90695cda9298a0663e667c12909246c358851
ensure_file $SHA1 $filename $URL

tar -xf $filename
cd $folder

./configure --prefix=$PYTHON_HOME --with-ensurepip=install

make -j${NPROC} && make install && echo "done"

$PYTHON_HOME/bin/pip3 install --upgrade pip
echo "pip upgraded"
