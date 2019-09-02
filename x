#!/bin/sh
set -e

./configure --build-tensorflow-ops
make -j8
