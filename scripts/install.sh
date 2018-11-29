#!/bin/sh
set -e

cd $(dirname $0)/..

pip3 install --no-index --user -U .

# TODO: static link in setup.py

./configure && make

USER_SITE_PKG=$(python3 -c "import sys; site_packages = [p for p in sys.path if 'site-packages' in p];print(site_packages[0]);")
PREFIX=$USER_SITE_PKG/kungfu

mkdir -p $PREFIX
install -v lib/*.so $PREFIX
