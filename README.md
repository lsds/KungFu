# KungFu

Easy, adaptive and fast distributed machine learning.

## Features

KungFu enables users to achieve *fast* and *adaptive* distributed machine learning. This is important because machine learning systems must cope with growing complex models and increasingly complicated deployment environments. KungFu has the following unique features:

* Simplicity: KungFu permits distributed training by adding only one line of code in the training program. KungFu is easy to deploy. It does not require partitioning resources as in parameter servers and heavy dependency like MPI in Horovod.
* Adaptive distributed training: KungFu provides many advanced [distributed optimizers](srcs/python/kungfu/optimizers/__init__.py) such as
communication-efficient [AD-PSGD](https://arxiv.org/abs/1710.06952) and small-batch-efficient [SMA](http://www.vldb.org/pvldb/vol12/p1399-koliousis.pdf) to help you address the cases in which [Synchronous SGD](https://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf) does not scale.
* Monitoring: KungFu supports [distributed SGD metrics](srcs/python/kungfu/optimizers/sync_sgd.py) such as [gradient variance](https://en.wikipedia.org/wiki/Variance) and [gradient noise scale](https://openai.com/blog/science-of-ai/) to help understand the training process with low overhead.
* Online control: KungFu provides control operators such as ``barrier`` and ``resize_cluster`` to seamlessly reconfigure training, even in response to monitored metrics.
* Extensibility: KungFu has a clean low-level API that allows an easy implementation of new distributed training, monitoring and control algorithms.

KungFu is fast and scalable. It exploits a high-performance implementation of communication, monitoring
and control operators, and adopts a decentralized architecture. Please check out the performance of KungFu in the Benchmark section below.

## Basic Usage

To use KungFu to scale out your TensorFlow training program, you simply need to make two changes:

1. Wrap the optimizer in ``SynchronousSGDOptimizer`` or another [distributed optimizer](srcs/python/kungfu/optimizers/__init__.py).

2. Run ``distributed_initializer()`` after calling ``global_variables_initializer()``.
    The distributed initializer synchronizes the initial variables on all workers.

```python
import tensorflow as tf
from kungfu.tensorflow.v1.optimizers import SynchronousSGDOptimizer

# Build model...
loss = ...

# Add KungFu Distributed Optimizer
opt = tf.train.AdamOptimizer(0.01)
opt = SynchronousSGDOptimizer(opt)

# Make training operation
train_op = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(opt.distributed_initializer()) # KungFu

    # Train your model for 10 steps.
    for step in range(10):
        sess.run(train_op)
```

See the [TensorFlow Session](examples/mnist_slp.py) and [TensorFlow Keras](examples/mnist_keras.py) examples for full training examples.

## Run

Download the MNIST dataset ([script](scripts/download-mnist.sh)) and run the following training script:

```bash
# Train a Single Layer Perception (SLP) model for the MNIST dataset using 4 CPUs for 10 data epochs.
kungfu-run -np 4 python3 examples/mnist_slp.py --data-dir=./mnist
```

If you want to run this example on two machines (each with 8 GPUs), run the following on both machines:

```bash
# Assume the machines have NIC eth0 and their IPs are 192.168.0.1 and 192.168.0.2.
# Assume NUM_GPU_SLOTS=8, NUM_GPUS=16
kungfu-run -np $NUM_GPUS \
    -H 192.168.0.1:$NUM_GPU_SLOTS,192.168.0.2:$NUM_GPU_SLOTS -nic eth0 \
    python3 examples/mnist_slp.py  --data-dir=./mnist
```

## Install

KungFu requires [Python 3](https://www.python.org/downloads/), [CMake 3.5+](https://cmake.org/install/), [Golang 1.11+](https://golang.org/dl/) and [TensorFlow <=1.13.2](https://www.tensorflow.org/install/pip#older-versions-of-tensorflow).

```bash
# Install tensorflow CPU
pip3 install tensorflow==1.13.1
# pip3 install tensorflow-gpu==1.13.1 # Using GPUs

# Download the KungFu source code
git clone https://github.com/lsds/KungFu.git

# Install KungFu
# export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) # Parallel build.
pip3 install .
```

KungFu provides ``kungfu-run`` to launch a training program on a multi-GPU server. Using the following command to build ``kungfu-run``.

```bash
# Build and install kungfu-run in the given GOBIN directory.
GOBIN=$(pwd)/bin go install -v ./srcs/go/cmd/kungfu-run

# Check if kungfu-run is built
./bin/kungfu-run -help
```

## Benchmark

We benchmark the performance of KungFu in a cluster that has 16 V100 GPUs hosted by 2 DGX-1 machines.
The machines are interconnected by a 100 Gbps network. We benchmark the training throughput of ResNet-50, VGG16 and InceptionV3. These models represent different kinds of training workloads.

In the synchronous training case, we compare KungFu (``SynchronousSGDOptimizer``) with [Horovod](https://github.com/horovod/horovod) (0.16.1). Horovod uses OpenMPI 4.0.0. We evaluate the spectrum of batch size (from 256 to 4096) commonly used by SGD users.
This batch size is evenly shared by the 16 GPUs.
KungFu outperforms Horovod on all tested models, in particular with small batch sizes which significantly raise the
frequency of synchronization.

![sync](benchmarks/system/result/sync-scalability.svg)

In the asynchronous training case, we compare KungFu (``PairAveragingOptimizer``) with TensorFlow parameter servers (1.13.1). We uses the same range of batch sizes as above. KungFu exhibits better scalability as well.

![async](benchmarks/system/result/async-scalability.svg)

All benchmark scripts are available [here](benchmarks/system/).

## Convergence

The synchronization algorithms (``SynchronousSGDOptimizer``, ``PairAveragingOptimizer`` and ``SynchronousAveragingOptimizer``)
can reach the same evaluation accuracy as Horovod. We validated this with the ResNet-50 and ResNet-101 models in the [TensorFlow benchmark](https://github.com/luomai/benchmarks/tree/cnn_tf_v1.12_compatible_kungfu).
You can also add your own KungFu distributed optimizer to the benchmark by adding one line of code, see [here](https://github.com/luomai/benchmarks/blob/1eb102a81cdcd42cdbea56d2d19f36a8018e9f80/scripts/tf_cnn_benchmarks/benchmark_cnn.py#L1197).

## Contribute

[Guideline](CONTRIBUTING.md)
