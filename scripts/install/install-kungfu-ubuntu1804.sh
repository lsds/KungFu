#!/bin/sh
# A reference script for installing KungFu on Ubuntu 18.04 
set -e

# Golang 1.11
sudo apt install software-properties-common
sudo add-apt-repository ppa:gophers/archive
sudo apt-get update
sudo apt-get install golang-1.11-go

echo "Add export PATH=$PATH:/usr/lib/go-1.11/bin to $HOME/.profile"
export PATH=$PATH:/usr/lib/go-1.11/bin

# CMake
sudo apt install cmake

# Python3
sudo apt install python3

# TensorFlow
pip3 install tensorflow==1.13.1

# KungFu
git clone git@github.com:lsds/KungFu.git
cd KungFu
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
pip3 install .
GOBIN=$(pwd)/bin go install -v ./srcs/go/cmd/kungfu-run/

# Test KungFu
sudo apt install curl
./scripts/download-mnist.sh
./bin/kungfu-run -np 4 -timeout 1h python3 examples/mnist_slp.py --n-epochs 10 --data-dir=mnist
