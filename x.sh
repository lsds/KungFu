#!/bin/sh
set -e

# flags() {
#     echo -q
#     echo -logdir logs
#     echo -np 2
# }

# export TF_CPP_MIN_LOG_LEVEL=1
# kungfu-run $(flags) python3 estimator-example.py

# ./configure --build-tensorflow-ops
# make -j 8

go install -v ./examples/go/...

batch_size=1000
kungfu-train-mnist-slp -bs $batch_size -epochs 10
