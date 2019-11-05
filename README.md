# KungFu

Easy and adaptive distributed machine learning at scale.

[![Build Status](https://travis-ci.com/lsds/KungFu.svg?branch=master)](https://travis-ci.com/lsds/KungFu)
[![Documentation Status](https://readthedocs.org/projects/kungfu/badge/?version=latest)](https://kungfu.readthedocs.io/en/latest/?badge=latest)

## Features

KungFu aims to help users achieve *fast* and *adaptive* distributed machine learning with *minimal* efforts. This is important because machine learning systems must cope with growing complex models and increasingly complicated deployment environments, making it
difficult to *empirically* configure a distributed training system that can constantly achieve high time-to-accuracy performance.
To address this, KungFu provides the following unique features:

* Simplicity: KungFu permits distributed training by adding minimal code in your training program. We make KungFu as easy as possible to install, deploy and run. It does not require extra deployment like parameter servers and heavy dependencies like MPI in Horovod.
* Adaptable distributed training: KungFu provides many advanced [distributed optimizers](srcs/python/kungfu/tensorflow/optimizers/__init__.py) such as
communication-efficient ``PairAveragingOptimizer`` and hyper-parameter-robust ``SynchronousAveragingOptimizer`` to help you address the cases in which conventional Synchronous SGD does not scale. See [Optimizers](https://github.com/lsds/KungFu#choosing-the-right-optimizer) for how to choose the right KungFu optimizer for your training scenario.
* Online monitoring and control: KungFu aims to support useful [distributed SGD metrics](srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py) such as [gradient noise scale](https://openai.com/blog/science-of-ai/) to help understand the training process with low overhead.
KungFu further provides control operators such as ``barrier`` and ``resize_cluster`` to help reconfigure training, even in response to monitored metrics.
* Fast and scalable: KungFu adopts a decentralized architecture and a non-blocking runtime. It exploits a high-performance implementation of communication, monitoring and control dataflow operators. Check out the performance of KungFu in [Benchmark](https://github.com/lsds/KungFu#benchmark).

We have been using KungFu for accelerating different kinds of deep learning models such as ResNet, OpenPose, BERT, CycleGAN and Alpha Zero. Check out their [examples](https://github.com/lsds/KungFu#examples).

## Usage

KungFu currently support TensorFlow and Keras. To scale out your TensorFlow program, for example, you needs to make two changes:

1. Wrap your ``tf.train.optimizer`` in KungFu's ``SynchronousSGDOptimizer``, ``SynchronousAveragingOptimizer``, ``PairAveragingOptimizer`` or another [distributed optimizer](srcs/python/kungfu/tensorflow/optimizers/__init__.py).

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
    from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
    sess.run(BroadcastGlobalVariablesOp())

    for step in range(10):
        sess.run(train_op)
```

Check the [Documentation](https://kungfu.readthedocs.io/en/latest/?badge=latest) for how to use KungFu with [Session](examples/tf1_mnist_session.py), [TensorFlow Keras](examples/tf1_mnist_keras.py), [Estimator](examples/tf1_mnist_estimator.py), and [GradientTape](examples/tf2_mnist_gradient_tape.py) in TensorFlow 1 and 2.
For KungFu with Keras, check out [here](examples/keras_mnist.py).

## Install

KungFu is implemented in Go and C++. It exposes a C interface in order to
allow an easy integration with existing deep learning systems.
Currently, it has Python binding for TensorFlow (including v1 and v2) and Keras (assuming you use TensorFlow as the backend).

KungFu for TensorFlow requires [Python 3](https://www.python.org/downloads/), [CMake 3.5+](https://cmake.org/install/), and [Golang 1.13+](https://golang.org/dl/).
KungFu has been tested with [TensorFlow](https://www.tensorflow.org/install/pip#older-versions-of-tensorflow) 1.12, 1.13, 1.15 and 2.0.0.
KungFu has a known installation issue with TensorFlow 1.14.
Assuming you have the above pre-requites, you can install KungFu as follows:

```bash
git clone https://github.com/lsds/KungFu.git
pip3 install KungFu/.
```

KungFu provides ``kungfu-run`` to launch a training program on a multi-GPU server.

```bash
# Build and install kungfu-run in the given GOBIN directory.
GOBIN=$(pwd)/KungFu/bin go install -v ./KungFu/srcs/go/cmd/kungfu-run

# Check if kungfu-run is built. You can export kungfu-run to your PATH in .bashrc
./KungFu/bin/kungfu-run -help
```

You can use KungFu with Docker. Check out the docker files for [GPU](docker/Dockerfile.tf-gpu) and [CPU](docker/Dockerfile.tf-cpu) machines.

## Run

We show how to run a KungFu program using a MNIST example.
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

## Examples

We have been using KungFu in training
different kinds of machine learning models.
We give a list of representative examples as follows:

* ***ImageNet - ResNet and other DNNs***: KungFu can speed up the training
of ResNet, VGG, DenseNet and others for ImageNet.
We show this in an [example](https://github.com/luomai/benchmarks/tree/cnn_tf_v1.12_compatible_kungfu/scripts/tf_cnn_benchmarks#running-kungfu) extended from the official [TensorFlow benchmark](https://github.com/luomai/benchmarks/tree/cnn_tf_v1.12_compatible_kungfu).

* ***Pose estimation - OpenPose***: Pose estimation models such as [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) are often batch-size sensitive.
We used KungFu in
a popular [OpenPose implementation](https://github.com/tensorlayer/openpose-plus) and achieved robust speed up in time-to-accuracy after
using the model averaging optimizers which preserves the merits of small batch training.

* ***Natural language processing - Google BERT***:
We have an example that shows how you can use few lines to enable distributed training for BERT. See the example [here](https://github.com/luomai/bert).

* ***Adversarial learning - CycleGAN***:
Generative adversarial networks (GANs) train multiple networks in parallel and often prefer using small batches for fast convergence. KungFu thus become an attractive option for GANs, because of its minimal changes to complex GAN programs
and new optimizers that decouple batch size and system parallelism.
Check out the simple [CycleGAN example](https://github.com/tensorlayer/cyclegan).

* ***Reinforcement learning - Alpha Zero***:
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

We have also run the same benchmark in a 16-server cluster (each has a P100).
KungFu exhibits better scalability in this communication-challenging environment,
and we thus only report the 16 V100 result here. You can find the benchmark scripts [here](benchmarks/system/).

## Choosing the right optimizer

KungFu aims to help users effectively decrease the
time to reach a desired accuracy (time-to-accuracy)
through scaling.
There are two ways to scaling the time-to-accuracy in KungFu:

* Synchronous SGD: Adopt parallel workers to improve the estimation of gradients, and expect to
reach a minima quickly using an increased learning rate.
* Model Averaging: Adopt parallel workers to explore a non-convex solution space and collaborate to find a good minima quickly.

***Synchronous SGD***:
Synchronous SGD is implemented as ``SynchronousSGDOptimizer`` in KungFu, equivalent to
the DistributedOptimizer in Horovod.
It requires users to carefully configure a distributed system
in order to address the challenges in scalability and accuracy.
Scalability-wise, all workers must exchange all gradients per iteration, making them difficult to scale in
a commodity cluster where bandwidth is limited and stragglers are common;
(ii) accuracy-wise, synchronous SGD couples training batch size with the number of workers,
and thus enforces users to use large batch sizes during scaling.
However, only a few models, such as ResNet and BERT, have been proven to converge fast
using large batch sizes, usually via [complex
model-specific hyper-parameter tuning](https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h5.pdf).
Without careful tuning, the test accuracy of a model
often suffers (see [paper](https://arxiv.org/abs/1609.04836) for more evidences).

***Model averaging***:
In model averaging,
each worker updates its local model using small batch size (thus the generality
of the found model is not affected), and constantly
exchange local training information to speed
up the search for minima. By decoupling
batch size with system parallelism, model averaging
is often *hyper-parameter robust*. We find
this property useful for distributed training users
as they often find it hard and expensive to
tune synchronous SGD at large scales.
In addition, model averaging is also shown
able to converge fast
with deep learning models (see [EA-SGD paper](https://arxiv.org/abs/1412.6651) and [Lookahead paper](https://arxiv.org/abs/1907.08610)).

Model averaging is implemented as ``SynchronousAveragingOptimizer`` and
``PairAveragingOptimizer`` in KungFu.
The former realizes the state-of-the-art model averaging algorithm ([SMA](http://www.vldb.org/pvldb/vol12/p1399-koliousis.pdf)); while the latter
implements an asynchronous model averaging algorithm ([AD-PSGD](https://arxiv.org/abs/1710.06952))
that is effective in an environment that has limited bandwidth and stragglers.

***Convergence evaluation***:
We have tested KungFu optimizers using ResNet-50 and ResNet-101 for ImageNet.
When using 8 V100, all KungFu optimizers together with Horovod
can all reach the target 75% accuracy.
When using 16 V100, Horovod and ``SynchronousSGDOptimizer`` suffer from
the increased batch size and their accuracy drop to 59% while
``SynchronousAveragingOptimizer`` and ``PairAveragingOptimizer`` still
reach the target 75%.
All these tests use a per-GPU batch size as 64 and other [hyper-parameters](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks#getting-started)
suggested by the TensorFlow benchmark authors.

***Writing your own optimizer***:
KungFu has a shared, scalable runtime that can support various
distributed training algorithms, even combining synchronous SGD and model averaging.
This runtime has an API to help write a custom distributed optimizer.

## Development

KungFu is designed with extensibility in mind.
It has a low-level API and a modular architecture, making
it suitable for implementing new distributed training, monitoring and control algorithms.
Check out the developer [guideline](CONTRIBUTING.md) for more information.
