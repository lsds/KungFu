# KungFu Artifact Evaluation OSDI 2020

This document describes how to evaluate the artifact
of the KungFu paper accepted by OSDI 2020. It
contains the information for the evaluation environment, the information for installing the KungFu library and relevant KungFu policy sample programs,
and the necessary scripts to re-run the experiments in the evaluation section of the paper.

## Paper

*KungFu: Making Training in Distributed Machine Learning Adaptive.*
Luo Mai, Guo Li, Marcel Wagenlander, Konstantinos Fertakis, Andrei-Octavian Brabete, Peter Pietzuch

Main contact for evaluation: Luo Mai (luo.mai@imperial.ac.uk)

## Preliminaries

The evaluation environment is hosted by a public cloud platform: Microsoft Azure. The base Virtual Machine (VM) image is `Canonical:UbuntuServer:18.04-LTS:latest` and you need to install
the following drivers and packages:

```bash
# Build utils
sudo add-apt-repository -y ppa:longsleep/golang-backports
sudo apt update
sudo apt install -y build-essential golang-go cmake python3-pip iperf nload htop

# CUDA 10.0
URL_PREFIX=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64
CUDA_REPO_PKG=cuda-repo-ubuntu1804_10.1.243-1_amd64.deb

wget -O /tmp/${CUDA_REPO_PKG} ${URL_PREFIX}/${CUDA_REPO_PKG}
sudo dpkg -i /tmp/${CUDA_REPO_PKG}
sudo apt-key adv --fetch-keys ${URL_PREFIX}/7fa2af80.pub

sudo apt update
sudo apt install -y cuda-drivers cuda cuda-10-0

# Cudnn 7
URL_PREFIX=https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64
CUDNN_REPO_PKG=nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

wget -O /tmp/${CUDNN_REPO_PKG} ${URL_PREFIX}/${CUDNN_REPO_PKG}
sudo dpkg -i /tmp/${CUDNN_REPO_PKG}
sudo apt-key adv --fetch-keys ${URL_PREFIX}/7fa2af80.pub

sudo apt update
sudo apt install -y libcudnn7 libcudnn7-dev


# Tensorflow
python3 -m pip install -U pip
pip3 install -U --user numpy==1.16 tensorflow-gpu==1.13.2
```

Once the VM is ready, you would need to install the KungFu library as follow:

```bash
git clone --branch osdi20-artifact https://github.com/lsds/KungFu.git
cd KungFu
pip3 install -U --user .
```

If you get the following warning:

```text
WARNING: The script kungfu-run is installed in '/home/user/.local/bin' which is not on PATH.
onsider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
```

Please add `kungfu-run` to your PATH:

```bash
PATH=$PATH:/home/user/.local/bin
```

**Important**: We provide a prepared VM for facilitating the above environment.
To gain a SSH access to such a VM, please contact the authors.

## Evaluation

We start with re-producing the performance benchmark result of KungFu. This benchmark depends on a synthetic ImageNet benchmark and incurs minimal dependency to hardware, real dataset and model implementation. It is thus most easy to re-produce.

### Monitoring Overhead (Figure 8)

In this experiment, we measure the overhead of computing online
monitored training metrics: (i) gradient noise scale and (ii) gradient variance. To run this experiment, you would need to
start a VM that has 4 K80 GPUs. In the paper, we present
the result on a dedicated DGX-1 with 8 V100 GPUs. We no longer
has the access to this machine.

You need to SSH to this VM and run the following command
to measure the overheads of computing **gradient noise scale**:

```bash
kungfu-run -np 4 python3 benchmarks/monitoring/benchmark.py --kf-optimizer=noise-scale --model=ResNet50 --batch-size=64 --interval==1
```

You would expect outputs like the below:

```text
...
[127.0.0.1.10003::stderr] Gradient Noise Scale: -1050.39917
[127.0.0.1.10000::stderr] Gradient Noise Scale: 6270.37646
[127.0.0.1.10001::stderr] Gradient Noise Scale: 996.805725
[127.0.0.1.10002::stderr] Gradient Noise Scale: 1410.21326
[127.0.0.1.10000::stdout] Iter #2: 48.7 img/sec per /gpu:0
[127.0.0.1.10001::stdout] Iter #2: 48.7 img/sec per /gpu:0
[127.0.0.1.10002::stdout] Iter #2: 48.8 img/sec per /gpu:0
[127.0.0.1.10003::stderr] Gradient Noise Scale: -907.974426
...
```

The same steps are applied to **gradient variance**.

```bash
kungfu-run -np 4 python3 benchmarks/monitoring/benchmark.py --kf-optimizer=variance --model=ResNet50 --batch-size=64 --interval==1
```

You would expect outputs like below:

```text
...
[127.0.0.1.10001::stderr] Variance: 0.000108428423
[127.0.0.1.10000::stderr] Variance: 0.000108428423
[127.0.0.1.10002::stderr] Variance: 0.000108428423
[127.0.0.1.10000::stdout] Iter #2: 47.4 img/sec per /gpu:0
[127.0.0.1.10001::stdout] Iter #2: 47.4 img/sec per /gpu:0
[127.0.0.1.10002::stdout] Iter #2: 47.4 img/sec per /gpu:0
[127.0.0.1.10003::stderr] Variance: 0.000108428423
[127.0.0.1.10003::stdout] Iter #2: 47.4 img/sec per /gpu:0
...
```

We then measure the optimal training throughput without monitoring
using the following command:





### Scalability (Figure 9)

In this experiment, we measure the scalability of the
asynchronous collective communication layer in KungFu.
We compare the ideal throughput (i.e., training throughput without
communication) and actual training throughput.

You will need to launch 8, 16, 32 VMs, respectively.
To measure the training performance
for ResNet50, on every VM, you need to run the following command:

[...]

You would expect the following outputs:

[...]

To measure the performance of MobileNetV2, you need to
run:

[...]

You would expect the following outputs:

[...]

### Scaling Performance (Figure 7)
