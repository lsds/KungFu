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

Download MNIST dataset ([script](scripts/azure/gpu-machine/download-mnist.sh)) and run the following training script.

```bash
python3 examples/mnist_mlp.py
```
