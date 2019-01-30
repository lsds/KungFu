#!/bin/sh
set -e

cd $HOME/kungfu-main

build(){
	git clean -fdx
	pip3 install --no-index --user -U .
	./scripts/go-install.sh --no-tests
}

build

#python3 benchmarks/tensorflow_synthetic_benchmark.py

np=4
H=127.0.0.1:4
ALGO=SIMPLE

./bin/kungfu-prun \
    -np $np \
    -H $H \
    -timeout 360s \
    -algo $ALGO \
    python3 \
    experiments/kungfu/kf_tensorflow_synthetic_benchmark.py
