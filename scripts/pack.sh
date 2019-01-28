#!/bin/sh
set -e

cd $(dirname $0)/..
[ -d gopath ] && rm -fr gopath
[ -d 3rdparty ] && rm -fr 3rdparty
if [ -d .git ]; then
    git clean -fdx
    git gc
fi

cd ..
[ -f KungFu.tar ] && rm KungFu.tar
[ -f KungFu.tar.bz2 ] && rm KungFu.tar.bz2
tar --exclude *.git -cf KungFu.tar KungFu
bzip2 KungFu.tar
du -hs KungFu.tar.bz2
