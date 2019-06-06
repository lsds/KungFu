# KungFu

KungFu distributed machine learning framework

## Install

Make sure you have `tensorflow` or `tensorflow-gpu` python library installed.

```bash
# install
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) # 4 seconds faster
pip3 install --no-index -U .

# FIXME: For Mac users, the following is required after the install:
# export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
```

To enable NCCL support

```bash
# uncomment to use your own NCCL
# export NCCL_HOME=$HOME/local/nccl

env \
    KUNGFU_ENABLE_NCCL=1 \
    pip3 install --no-index --user -U .
```

## Example

Download MNIST dataset ([script](scripts/download-mnist.sh)) and run the following training script.

```bash
python3 examples/mnist_mlp.py
```

## Build for release

```bash
# build a .whl package for release
pip3 wheel -vvv --no-index .
```
