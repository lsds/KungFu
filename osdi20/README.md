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
git clone https://github.com/lsds/KungFu.git
cd KungFu
pip3 install -U --user .
```

Different experiments may have specific dependency to dataset, policy programs and scripts. Please refer to the corresponding
sub-sections below.

**Note**: We provide a prepared VM for facilitating the reproduction.
To gain a SSH access to this VM, please contact the authors.

## Evaluation

We start with re-producing the performance benchmark result of KungFu. This benchmark depends on a synthetic ImageNet benchmark and incurs minimal dependency to hardware, real dataset and model implementation. It is thus most easy to re-produce.

### Monitoring Overhead (Figure 8)

In this experiment, we measure the overhead of computing online
monitored training metrics: (i) gradient noise scale and (ii) gradient variance. To run this experiment, you would need to
start a VM that has 4 P100 GPUs. In the paper, we present
the result on a dedicated DGX-1 with 8 V100 GPUs. We no longer
has the access to this machine.

You then SSH to this VM and run the following command
to measure the overheads of monitoring **gradient noise scale**:

[...]

You would expect the following outputs:

[...]

Based on these outputs, we can calculate the overheads
of monitoring is [...].

The same steps are applied to **gradient variance**.

[...]

You would expect the following outputs:

[...]

### Scalability (Figure 9)
