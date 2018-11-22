#!/bin/sh
set -e

PREFIX=$HOME/code/mirror

mkdir -p $PREFIX/github.com/tensorflow

[ ! -d $PREFIX/github.com/tensorflow/benchmarks ] && git clone https://github.com/tensorflow/benchmarks.git $PREFIX/github.com/tensorflow/benchmarks

cd $PREFIX/github.com/tensorflow/benchmarks
cd scripts/tf_cnn_benchmarks

python3 tf_cnn_benchmarks.py --num_gpus=4 --batch_size=32 --model=resnet50 --variable_update=parameter_server
