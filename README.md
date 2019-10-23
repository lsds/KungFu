# KungFu

Adaptive distributed machine learning.

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

### Mac Users

For Mac users, the following is required after the install:

```bash
export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
```

## Example

Download MNIST dataset ([script](scripts/download-mnist.sh)) and run the following training script.

```bash
# Download the MNIST dataset in a ./mnist folder in the current directory.
./scripts/download-mnist.sh

# Train a Single Layer Perception (SLP) model for the MNIST dataset using 4 CPUs for 10 data epochs.
./bin/kungfu-run -np 4 -timeout 1h python3 examples/mnist_slp.py --n-epochs 10
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

### Use NVIDIA NCCL

KungFu can use [NCCL](https://developer.nvidia.com/nccl) to leverage GPU-GPU direct communication.
However, the use of NCCL enforces KungFu to serialize the execution of all-reduce operations, which can hurt performance.

```bash
# uncomment to use your own NCCL
# export NCCL_HOME=$HOME/local/nccl

KUNGFU_USE_NCCL=1 pip3 install .
```
