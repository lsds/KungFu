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
[ -f kungfu.tar ] && rm kungfu.tar
[ -f kungfu.tar.bz2 ] && rm kungfu.tar.bz2
tar --exclude *.git -cf kungfu.tar kungfu
bzip2 kungfu.tar
du -hs kungfu.tar.bz2
