# KungFu

Easy, adaptive and fast distributed machine learning.

[![Build Status](https://travis-ci.com/lsds/KungFu.svg?branch=master)](https://travis-ci.com/lsds/KungFu)
[![Documentation Status](https://readthedocs.org/projects/kungfu/badge/?version=latest)](https://kungfu.readthedocs.io/en/latest/?badge=latest)

## Features

KungFu enables users to achieve *fast* and *adaptive* distributed machine learning. This is important because machine learning systems must cope with growing complex models and increasingly complicated deployment environments. KungFu has the following unique features:

* Simplicity: KungFu permits distributed training by adding minimal code in your training program. KungFu is easy to deploy and run, because it does not require extra deployment like parameter servers and heavy dependencies like MPI in Horovod.
* Adaptable distributed training: KungFu provides many advanced [distributed optimizers](srcs/python/kungfu/tensorflow/v1/optimizers/__init__.py) such as
communication-efficient ``PairAveragingOptimizer`` and hyper-parameter-robust ``SynchronousAveragingOptimizer`` to help you address the cases in which conventional Synchronous SGD does not scale. See [Optimizers](https://github.com/lsds/KungFu#optimizers) for how to choose the right KungFu optimizer for your training scenario.
* Online monitoring and control: KungFu supports [distributed SGD metrics](srcs/python/kungfu/tensorflow/v1/optimizers/sync_sgd.py) such as [gradient variance](https://en.wikipedia.org/wiki/Variance) and [gradient noise scale](https://openai.com/blog/science-of-ai/) to help understand the training process with low overhead.
KungFu further provides control operators such as ``barrier`` and ``resize_cluster`` to seamlessly reconfigure training, even in response to monitored metrics.
* Fast and scalable: KungFu adopts a decentralized architecture and exploits a high-performance implementation of communication, monitoring and control operators. Check out the performance of KungFu in the [Benchmark](https://github.com/lsds/KungFu#benchmark).

KungFu is extensible. It has a clean low-level API that allows an easy implementation of new distributed training, monitoring and control algorithms.

## Usage

To scale out your TensorFlow training program, you simply need to make two changes:

1. Wrap your ``tf.train.optimizer`` in KungFu's ``SynchronousSGDOptimizer``, ``SynchronousAveragingOptimizer``, ``PairAveragingOptimizer`` or another [distributed optimizer](srcs/python/kungfu/tensorflow/v1/optimizers/__init__.py).

2. Ensure all distributed workers start with consistent states by broadcasting a worker's global variables.

```python
import tensorflow as tf

# Build model...
loss = ...
opt = tf.train.AdamOptimizer(0.01)

# KungFu Step 1: Wrap tf.optimizer in KungFu optimizers
from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
opt = SynchronousSGDOptimizer(opt)

# Make training operation
train_op = opt.minimize(loss)

# Train your model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # KungFu Step 2: ensure distributed workers start with consistent states
    from kungfu.tensorflow.v1.initializer import BroadcastGlobalVariablesOp
    sess.run(BroadcastGlobalVariablesOp())

    for step in range(10):
        sess.run(train_op)
```

See TensorFlow full training examples that use [Session](examples/tf1_mnist_session.py), [Keras](examples/tf1_mnist_keras.py) and [Estimator](examples/tf1_mnist_estimator.py), respectively.

## Install

KungFu is implemented in Go and C++. It exposes a C interface so that it can be easily integrated within existing machine learning systems.
Currently, it has a Python binding for TensorFlow.

KungFu for TensorFlow requires [Python 3](https://www.python.org/downloads/), [CMake 3.5+](https://cmake.org/install/), [Golang 1.13+](https://golang.org/dl/) and [TensorFlow <=1.13.2](https://www.tensorflow.org/install/pip#older-versions-of-tensorflow).
It can be installed with the following few lines, assuming you have the above pre-requites.

```bash
# Download the KungFu source code
git clone https://github.com/lsds/KungFu.git

# Install KungFu
# export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) # Parallel build.
pip3 install .
```

KungFu provides ``kungfu-run`` to launch a training program on a multi-GPU server.

```bash
# Build and install kungfu-run in the given GOBIN directory.
GOBIN=$(pwd)/bin go install -v ./srcs/go/cmd/kungfu-run

# Check if kungfu-run is built
./bin/kungfu-run -help
```

You can use KungFu with Docker. Check out the docker files for [GPU](docker/Dockerfile.tf-gpu) and [CPU](docker/Dockerfile.tf-cpu) machines.

## Examples

### MNIST

Download the MNIST dataset ([script](scripts/download-mnist.sh)) and run the following training script:

```bash
# Train a Single Layer Perception (SLP) model for the MNIST dataset using 4 CPUs for 10 data epochs.
kungfu-run -np 4 python3 examples/tf1_mnist_session.py --data-dir=./mnist
```

If you want to run this example on two machines (each with 8 GPUs), run the following on both machines:

```bash
# Assume the machines have NIC eth0 and their IPs are 192.168.0.1 and 192.168.0.2.
# Assume NUM_GPU_SLOTS=8, NUM_GPUS=16
kungfu-run -np $NUM_GPUS \
    -H 192.168.0.1:$NUM_GPU_SLOTS,192.168.0.2:$NUM_GPU_SLOTS -nic eth0 \
    python3 examples/tf1_mnist_session.py  --data-dir=./mnist
```

``kungfu-run`` use the ``nic`` option to infer its IP and thus its role in the cluster.

### ImageNet

KungFu also has a ImageNet [example](https://github.com/luomai/benchmarks/tree/cnn_tf_v1.12_compatible_kungfu/scripts/tf_cnn_benchmarks#running-kungfu) which is slightly modified from the [TensorFlow benchmark](https://github.com/luomai/benchmarks/tree/cnn_tf_v1.12_compatible_kungfu).
You can add your own KungFu distributed optimizer to the ImageNet example by adding one line of code, see [here](https://github.com/luomai/benchmarks/blob/cnn_tf_v1.12_compatible_kungfu/scripts/tf_cnn_benchmarks/benchmark_cnn.py#L1198).

### BERT

We have an example that shows how you can use a very few lines to enable distributed training for Google BERT using KungFu. See the example [here](https://github.com/luomai/bert).

### Alpha Zero

We are working on an Alpha Zero distributed training example and will release it soon.

## Benchmark

We benchmark KungFu in a cluster that has 16 V100 GPUs hosted by 2 DGX-1 machines.
The machines are interconnected by a 100 Gbps network. We measure the training throughput of ResNet-50, VGG16 and InceptionV3. These models represent different kinds of training workloads.

In the synchronous training case, we compare KungFu (``SynchronousSGDOptimizer``) with [Horovod](https://github.com/horovod/horovod) (0.16.1). Horovod uses OpenMPI 4.0.0. We evaluate the spectrum of batch size (from 256 to 4096) commonly used by SGD users.
This batch size is evenly shared by the 16 GPUs.
KungFu outperforms Horovod on all tested models, in particular with small batch sizes which significantly raise the
frequency of synchronization.

![sync](benchmarks/system/result/sync-scalability.svg)

In the asynchronous training case, we compare KungFu (``PairAveragingOptimizer``) with TensorFlow parameter servers (1.13.1). We uses the same range of batch sizes as above. KungFu exhibits better scalability as well.

![async](benchmarks/system/result/async-scalability.svg)

All benchmark scripts are available [here](benchmarks/system/).

## Optimizers

KungFu provide many useful distributed optimizers that achieve different trade-off between hardware efficiency and statistical efficiency.

* ``SynchronousSGDOptimizer``: this optimizer is the most common one. However, it inevitably increases batch size
when scaling out training, and reduces the number of model updates,
which eventually adversely affect the generality of the trained model ([paper](https://arxiv.org/abs/1609.04836)).
The loss in the updates must be compensated by many [hyper-parameter tuning](https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h5.pdf)
which is unfortunately model-specific.
Synchronous SGD is also hard to scale within a commodity network because it
maintains a global barrier and transmit all gradients.
* ``SynchronousAveragingOptimizer``: this optimizer adopts [synchronous model averaging](http://www.vldb.org/pvldb/vol12/p1399-koliousis.pdf).
It is proven to converge in this [paper](https://arxiv.org/abs/1412.6651).
It often converges faster than the synchronous SGD as it allows parallel workers
to slightly diverge, which improves the generality of the trained model. Hence, the
hyper-parameters you find useful in a single node scenario is often also
effective in a parallel training case, making this optimizer hyper-parameter-robust.
* ``PairAveragingOptimizer``: this optimizer asynchronously synchronizes workers, making
it suitable for the environment that has limited bandwidth
and stragglers. This optimizer is also proven to converge with deep learning models in this [paper](https://arxiv.org/abs/1710.06952).

We have tested these optimizers using ResNet-50 and ResNet-101 in the TensorFlow benchmark.
When using 8 V100, all KungFu optimizers together with Horovod
can all reach the target 75% accuracy.
When using 16 V100, Horovod and ``SynchronousSGDOptimizer`` suffer from the
loss in model updates and their accuracy drop to 59% while
``SynchronousAveragingOptimizer`` and ``PairAveragingOptimizer`` still
reach the target 75%.
All these convergence tests are using the same [hyper-parameter setup](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks#getting-started)
along with a per-GPU batch size as 64, suggested by the TensorFlow benchmark authors.

## Contribute

[Guideline](CONTRIBUTING.md)
