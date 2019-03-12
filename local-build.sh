#!/bin/sh
set -e

cd $(dirname $0)
. ./scripts/utils/measure.sh

./configure --build-tensorflow-ops

measure make
