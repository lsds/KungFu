#!/bin/sh
set -e

cd $(dirname $0)/..
git clean -fdx
[ -d gopath ] && rm -fr gopath
[ -d 3rdparty ] && rm -fr 3rdparty
git gc

cd ..
[ -f kungfu.tar ] && rm kungfu.tar
[ -f kungfu.tar.bz2 ] && rm kungfu.tar.bz2
tar --exclude *.git -cf kungfu.tar kungfu
bzip2 kungfu.tar
du -hs kungfu.tar.bz2
