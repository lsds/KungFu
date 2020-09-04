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
PATH=$PATH:$HOME/.local/bin
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
have access to this machine.

You need to SSH to this VM. We first measure
the training throughput of each GPU without monitoring:

```bash
kungfu-run -np 4 python3 benchmarks/monitoring/benchmark.py --kf-optimizer=sync-sgd --model=ResNet50 --batch-size=64
```

You would expect outputs like the below:

```text
...
[127.0.0.1.10002::stdout] Iter #0: 49.3 img/sec per /gpu:0
[127.0.0.1.10000::stdout] Iter #0: 49.4 img/sec per /gpu:0
[127.0.0.1.10001::stdout] Iter #0: 49.3 img/sec per /gpu:0
[127.0.0.1.10003::stdout] Iter #0: 49.3 img/sec per /gpu:0
...
```

To measure the overheads of computing **gradient noise scale**,
we need to switch the `kf-optimizer` to `noise-scale` like follow:

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

This shows that the training throughput drops from 49.2 to 48.7
with extra gradient noise scale computation.

The same measurement is applied to **gradient variance** monitoring:

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

This shows that the training throughput drops from 49.2 to 47.4
with extra gradient variance computation.

### Scalability (Figure 9)

In this experiment, we measure the scalability of the
asynchronous collective communication layer in KungFu.
We compare the ideal throughput (i.e., training throughput without
communication) and actual training throughput.

You will need to launch a cluster that has `N` VMs.
For simplicity, assuming we have 2 VMs with the private IPs: `10.0.0.19` and `10.0.0.20`.
These IPs are bind with the NIC: `eth0`.
You can launch the 2-VM data parallel training by running the following command on **each** VM.

```bash
kungfu-run -np 2 -H 10.0.0.19:1,10.0.0.20:1 -nic=eth0 python3 benchmarks/system/benchmark_kungfu.py --kf-optimizer=sync-sgd --model=ResNet50 --batch-size=64
```

You would expect outputs like below in the end on one of the VMs:

```text
...
[10.0.0.19.10000::stdout] Running benchmark...
[10.0.0.19.10000::stdout] Iter #0: 50.5 img/sec per /gpu:0
[10.0.0.19.10000::stdout] Iter #1: 50.5 img/sec per /gpu:0
[10.0.0.19.10000::stdout] Iter #2: 50.4 img/sec per /gpu:0
[10.0.0.19.10000::stdout] Iter #3: 50.1 img/sec per /gpu:0
[10.0.0.19.10000::stdout] Iter #4: 50.1 img/sec per /gpu:0
[10.0.0.19.10000::stdout] Iter #5: 50.0 img/sec per /gpu:0
[10.0.0.19.10000::stdout] Iter #6: 50.1 img/sec per /gpu:0
[10.0.0.19.10000::stdout] Iter #7: 49.9 img/sec per /gpu:0
[10.0.0.19.10000::stdout] Iter #8: 49.7 img/sec per /gpu:0
[10.0.0.19.10000::stdout] Iter #9: 49.8 img/sec per /gpu:0
[10.0.0.19.10000::stdout] Img/sec per /gpu:0: 50.1 +-0.5
[10.0.0.19.10000::stdout] RESULT: 50.118283 +-0.494631 {"framework":"kungfu","np":2,"strategy":"BINARY_TREE_STAR","bs":64,"model":"ResNet50","xla":false,"kf-opt":"sync-sgd","fuse":false,"nvlink":"false"}
[I] all 1/2 local peers finished, took 2m40.691635844s
```

To run the scalability experiment using another model: `MobileNetV2`, you
only need to replace the `--model=ResNet50` with `--model=MobileNetV2`.

The same operation is applied to cluster with any number (i.e.,
8, 16, 32, ...) of VMs.

### Scaling Performance (Figure 7)
