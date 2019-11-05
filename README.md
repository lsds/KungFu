# KungFu

Easy, fast and adaptive distributed machine learning.

[![Build Status](https://travis-ci.com/lsds/KungFu.svg?branch=master)](https://travis-ci.com/lsds/KungFu)
[![Documentation Status](https://readthedocs.org/projects/kungfu/badge/?version=latest)](https://kungfu.readthedocs.io/en/latest/?badge=latest)

## Features

KungFu aims to help users achieve *fast* and *adaptive* distributed machine learning with minimal efforts. This is important because machine learning systems must cope with growing complex models and increasingly complicated deployment environments. KungFu has the following unique features:

* Simplicity: KungFu permits distributed training by adding minimal code in your training program. KungFu is easy to deploy and run, because it does not require extra deployment like parameter servers and heavy dependencies like MPI in Horovod.
* Adaptable distributed training: KungFu provides many advanced [distributed optimizers](srcs/python/kungfu/tensorflow/optimizers/__init__.py) such as
communication-efficient ``PairAveragingOptimizer`` and hyper-parameter-robust ``SynchronousAveragingOptimizer`` to help you address the cases in which conventional Synchronous SGD does not scale. See [Optimizers](https://github.com/lsds/KungFu#optimizers) for how to choose the right KungFu optimizer for your training scenario.
* Online monitoring and control: KungFu supports [distributed SGD metrics](srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py) such as [gradient variance](https://en.wikipedia.org/wiki/Variance) and [gradient noise scale](https://openai.com/blog/science-of-ai/) to help understand the training process with low overhead.
KungFu further provides control operators such as ``barrier`` and ``resize_cluster`` to seamlessly reconfigure training, even in response to monitored metrics.
* Fast and scalable: KungFu adopts a decentralized architecture and exploits a high-performance implementation of communication, monitoring and control operators. Check out the performance of KungFu in the [Benchmark](https://github.com/lsds/KungFu#benchmark).

We have been using KungFu for accelerating different kinds of deep learning models such as ResNet, OpenPose, BERT, and CycleGAN. Check out their [examples](https://github.com/lsds/KungFu#examples).

## Usage

KungFu can be used with TensorFlow and Keras. To scale out your TensorFlow training program, for example, you simply need to make two changes:

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

Check the documentation for more details regarding how to use KungFu with [Session](examples/tf1_mnist_session.py), [TensorFlow Keras](examples/tf1_mnist_keras.py), [Estimator](examples/tf1_mnist_estimator.py), and [GradientTape](examples/tf2_mnist_gradient_tape.py) in TensorFlow 1 and 2.
For KungFu with Keras, check out this [example](examples/keras_mnist.py).

## Install

KungFu is implemented in Go and C++. It exposes a C interface in order to
allow an easy integration with existing machine learning systems.
Currently, it has Python binding for TensorFlow (including v1 and v2) and Keras (assuming you use TensorFlow as the backend).

KungFu for TensorFlow requires [Python 3](https://www.python.org/downloads/), [CMake 3.5+](https://cmake.org/install/), and [Golang 1.13+](https://golang.org/dl/).
KungFu has been tested with [TensorFlow 1.12, 1.13, 1.15 and 2.0.0](https://www.tensorflow.org/install/pip#older-versions-of-tensorflow).
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

### ImageNet - ResNet and other DNNs

KungFu can speed up the training
of ResNet, VGG, DenseNet and others for ImageNet.
We show this in an [example](https://github.com/luomai/benchmarks/tree/cnn_tf_v1.12_compatible_kungfu/scripts/tf_cnn_benchmarks#running-kungfu) extended from the official [TensorFlow benchmark](https://github.com/luomai/benchmarks/tree/cnn_tf_v1.12_compatible_kungfu).

### Pose estimation - OpenPose

Pose estimation models such as [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) are often batch-size sensitive.
We used KungFu in
a popular [OpenPose implementation](https://github.com/tensorlayer/openpose-plus) and achieved robust speed up in time-to-accuracy after
using the model averaging optimizers which preserves small batch training during scaling.

### Natural language processing - BERT

BERT is an emerging NLP training workload. We have an example that shows how you can use few lines to enable distributed training for BERT. See the example [here](https://github.com/luomai/bert).

### Generative adversarial networks - CycleGAN

Generative adversarial networks (GAN) train multiple networks in parallel and are often limited to use
small batches for convergence. KungFu thus become an attractive option for GANs, because of its minimal changes to complex GAN programs
and new distributed optimizers that decouple batch size and system parallelism.
Check out the simple [CycleGAN example](https://github.com/tensorlayer/cyclegan).

### Reinforcement learning - Alpha Zero

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

## Development

KungFu is highly extensible. It has a clean low-level API that allows an easy implementation of new distributed training, monitoring and control algorithms.
Check the developer [guideline](CONTRIBUTING.md) for more information.
