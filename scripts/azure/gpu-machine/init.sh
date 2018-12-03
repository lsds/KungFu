#!/bin/sh
set -e

cd $(dirname $0)
./install-cuda-9.sh
./install-cudnn-7.sh
./install-tensorflow-gpu.sh

sudo apt install -y build-essential cmake
./install-golang1.11.sh
