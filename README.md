# KungFu

Adaptive distributed machine learning.

## Features

TODO

## Usage

To use KungFu, make the following additions to your program. This example uses TensorFlow.

1. Wrap optimizer in ``kungfu.optimizers.SyncSGDOptimizer`` or other KungFu [distributed optimizers](srcs/python/kungfu/optimizers/__init__.py).

2. Run ``sess.run(kungfu_optimizer.distributed_initializer())`` after you call ``sess.run(tf.global_variables_initializer())``.
    The distributed initializer will automatically synchronise the initial variables on all KungFu workers based on the chosen distributed optimizer.

Example (see the [example](examples/mnist_slp.py) for a full training example):

```python
import tensorflow as tf
import kungfu as kf

# Build model...
loss = ...

# You may want to scale the learning rate
opt = tf.train.AdagradOptimizer(0.01 * kf.ops.current_cluster_size())

# Add KungFu Distributed Optimizer
opt = kf.optimizers.SyncSGDOptimizer(opt)

# Make training operation
train_op = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(kungfu_optimizer.distributed_initializer()) # KungFu

    # Train your model for 10 steps.
    for step in range(10):
        sess.run(train_op)
```

## Run

Download MNIST dataset ([script](scripts/download-mnist.sh)) and run the following training script.

```bash
# Train a Single Layer Perception (SLP) model for the MNIST dataset using 4 CPUs for 10 data epochs.
./bin/kungfu-run -np 4 python3 examples/mnist_slp.py --data-dir=./mnist
```

If you want to run this example on two machines (assuming each machine 8 GPUs), run the following command:

```bash
# Assuming the machines have the following IPs: 192.168.0.1 and 192.168.0.2.
NUM_GPU_SLOTS=8
NUM_GPUS=16
./bin/kungfu-run -np $NUM_GPUS -H 192.168.0.1:$NUM_GPU_SLOTS,192.168.0.2:$NUM_GPU_SLOTS python3 examples/mnist_slp.py  --data-dir=./mnist
```

## Install

KungFu requires [Python 3](https://www.python.org/downloads/), [CMake 3](https://cmake.org/install/), [Golang 1.11+](https://golang.org/dl/) and [TensorFlow <=1.13.2](https://www.tensorflow.org/install/pip#older-versions-of-tensorflow).

```bash
# Install tensorflow CPU
pip3 install tensorflow==1.13.1
# pip3 install tensorflow-gpu==1.13.1 # Using GPUs

# Download the KungFu source code
git clone https://github.com/lsds/KungFu.git

# Install KungFu
# export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) # Parallel install.
pip3 install .
```

KungFu provides: *kungfu-run*, similar to [mpirun](https://horovod.readthedocs.io/en/latest/mpirun.html), to launch a TensorFlow program on multiple GPU/CPU devices in a server.
Using the following command to build kungfu-run.

```bash
# Build kungfu-run in the given GOBIN directory.
GOBIN=$(pwd)/bin go install -v ./srcs/go/cmd/kungfu-run/

# Check if kungfu-run is built
./bin/kungfu-run -help
```

For Mac users, the following is required after the install:

```bash
export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
```

## Benchmark

TODO

## Contribute

[Contributor Guideline](CONTRIBUTING.md).
