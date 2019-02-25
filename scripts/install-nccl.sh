#!/bin/sh
set -e

mkdir -p $HOME/tmp
cd $HOME/tmp

ensure_file() {
    local sha1=$1
    local filename=$2
    local URL=$3

    if [ -f $filename ]; then
        if [ $(sha1sum $filename | awk '{print $1}') != $sha1 ]; then
            "$filename has invalid sha1 sum"
            rm $filename
        else
            echo "use existing file $filename"
            return
        fi
    fi

    curl -vLOJ $URL
    if [ ! -f $filename ]; then
        echo "Download $filanem failed"
        return 1
    fi
    if [ $(sha1sum $filename | awk '{print $1}') != $sha1 ]; then
        echo "downloaded file has invalid sha1"
        return 1
    fi
}

version=2.4.2-1
filename=nccl-${version}.tar.gz
URL=https://github.com/NVIDIA/nccl/archive/v${version}.tar.gz
folder=${filename%.tar.gz}

ensure_file 0dcac5994c54af839e2e1646e88d12c2fe77b338 $filename $URL
tar -xf $filename
cd $folder

export PREFIX=$HOME/local/nccl
make -j$(nproc)
make install
