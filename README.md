# KungFu

KungFu distributed machine learning framework

## Build

Make sure you have tensorflow python installed.

```bash
# build a .whl package for release
pip3 wheel -vvv --no-index .

# install
pip3 install --no-index -U .
```

## Example

Download MNIST dataset ([script](scripts/download-mnist.sh)) and run the following training script.

```bash
# FIXME: For Mac users, the following is required
# export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
python3 examples/mnist_mlp.py
```
