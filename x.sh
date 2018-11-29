#!/bin/sh
set -e

./examples/kungfu-train.py 2>&1 | tee single.log
./scripts/parallel-train.sh 2>&1 | tee parallel.log
