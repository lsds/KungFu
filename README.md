# KungFu

High-performance, adaptive, distributed machine learning.

## Install

KungFu has pre-requisites of Python 3, [Golang](https://golang.org/dl/) and [TensorFlow 1.x](https://www.tensorflow.org/install/pip#older-versions-of-tensorflow).

```bash
# install tensorflow CPU
pip3 install tensorflow==1.13.1
# pip3 install tensorflow-gpu==1.13.1 # Using GPUs

# downaload the KungFu source code
git clone https://github.com/lsds/KungFu.git

# install KungFu
# export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) # Parallel install.
pip3 install .
```

KungFu uses: *kungfu-prun*, similar to [mpirun](https://horovod.readthedocs.io/en/latest/mpirun.html), to launch a TensorFlow program on multiple GPU/CPU devices on a local server.
Using the following command to build kungfu-prun.

```bash
# Build kungfu-prun in the current ./bin/ directory.
./configure --build-tools
make

# Check if kungfu-prun is built
./bin/kungfu-prun -help
```

### (Optional) Mac Users

For Mac users, the following is required after the install:

```bash
export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
```

### (Optional) NVIDIA NCCL Support

KungFu can use NCCL to accelerate GPU-GPU communication.

```bash
# uncomment to use your own NCCL
# export NCCL_HOME=$HOME/local/nccl

env \
    KUNGFU_USE_NCCL=1 \
    pip3 install --no-index --user -U .
```

## Example

Download MNIST dataset ([script](scripts/download-mnist.sh)) and run the following training script.

```bash
# Download the MNIST dataset in a mnist folder in the current directory.
./scripts/download-mnist.sh

# Train the mnist_slp program using 4 CPUs.
./bin/kungfu-prun -np 4 -timeout 1h python3 examples/mnist_slp.py
```

## Contribution

### Format code

```bash
./scripts/clean-code.sh --fmt-py
```

### Build for release

```bash
# build a .whl package for release
pip3 wheel -vvv --no-index .
```
