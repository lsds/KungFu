# KungFu Contributor Guide

## Intro

KungFu is a distributed machine learning framework implemented in Golang and C++.
It provides Python binding for TensorFlow.

KungFu can be integrated into existing TensorFlow programs as easy as Horovod.
It can be used as a standalone collective communication library for C/C++/Golang programs, similar to MPI.

## Requirements

* Golang1.11 or above is required (For the [Modules](https://github.com/golang/go/wiki/Modules) feature).
* CMake is required for building the C++ sources.
* Python and TensorFlow is required if you are going to build the TensorFlow binding.
* gtest is used for unittest, it can be auto fetched and built from source.

## Project Structure

All source code are under `./srcs/<lang>/` where `<lang> := cpp | go | python`.

### Libraries

* libkungfu-comm: a C library implemented in Golang
* libkungfu: the public C/C++ library based on kungfu-comm

### Concepts

* Peer: the basic unit in a KungFu cluster. A **Peer** usually represents a system process.

* PeerID: **PeerID** is the unique identifier of a **Peer**. It tells the KungFu runner how to start the **Peer** and let peers to locate each other. **PeerID** is immutable during the life cycle of a **Peer**.

* Session: a group of **Peer**s interconnected by **Connection**s. Currently a **Peer** can be in at most one **Session** at the same time. **Peer**s in a **Session** can perform collective commuication operations, such as _allreduce_, _broadcast_.

* PeerList: a **PeerList** is an ordered list of **PeerID** from all peers in a **Session**.
  All **Peer**s in the same **Session** have the same **PeerList**, which allows Rank to be defined.

* Connection: the communication channel between two **Peer**s. A **Connection** is the high-level abstraction of a network connection, usually TCP.

* Message: the basic communication element in a **Connection**.

* HostSpec: HostSpec is the metadata that describes a host machine.

* Graph: A directed communication graph of peers. A graph may contain self loops. The vertices are numbered from 0 to n - 1.

### Components

* Server: A TCP-based server that accepts **Connection**s

* Client: A TCP-based client that can send data using **Connection**s

* Handler: A TCP-based connection handler that can handle **Connection**s

## Useful commands

### Format code

```bash
./scripts/clean-code.sh --fmt-py
```

### Build for release

```bash
# build a .whl package for release
pip3 wheel -vvv --no-index .
```

### Docker

```bash
# Run the following command in the KungFu folder
docker build -f docker/Dockerfile.tf-gpu -t kungfu:gpu .

# Run the docker
docker run -it kungfu:gpu
```

## Use NVIDIA NCCL

KungFu can use [NCCL](https://developer.nvidia.com/nccl) to leverage GPU-GPU direct communication.

```bash
# uncomment to use your own NCCL
# export NCCL_HOME=$HOME/local/nccl

KUNGFU_ENABLE_NCCL=1 pip3 install .
```

To use NVLink, add the following to your Python code

```python
...

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from kungfu.python import _get_cuda_index
config.gpu_options.visible_device_list = str(_get_cuda_index())

...

with tf.Session(config=config) as sess:

...

```

and add `-allow-nvlink` flag to `kungfu-run` command

```bash
# export NCCL_DEBUG=INFO # uncomment to enable
kungfu-run -np 4 -allow-nvlink python3 benchmarks/system/benchmark_kungfu.py
```


## Debug

```bash
export KUNGFU_CONFIG_LOG_LEVEL=DEBUG # or INFO | WARN | ERROR, the default is INFO
```
