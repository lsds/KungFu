# KungFu Artifact Evaluation OSDI 2020

This document describes how to evaluate the artifacts of the KungFu paper, which has been accepted for publication at the OSDI 2020 conference. It
contains information about the evaluation environment, the installation of the KungFu library and relevant KungFu policy sample programs,
and the necessary scripts to re-run the experiments from the evaluation section of the paper.

## Paper

*KungFu: Making Training in Distributed Machine Learning Adaptive.*
Luo Mai, Guo Li, Marcel Wagenlander, Konstantinos Fertakis, Andrei-Octavian Brabete, Peter Pietzuch (Imperial College London)

## Preliminaries

The evaluation environment is hosted on a public cloud platform: Microsoft Azure. The base Virtual Machine (VM) image is `Canonical:UbuntuServer:18.04-LTS:latest`, and you need to install the following drivers and packages:

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

Once the VM is ready, you need to install KungFu as follows:

```bash
git clone --branch osdi20-artifact https://github.com/lsds/KungFu.git
cd KungFu
pip3 install -U --user .
```

If you get the following warning:

```text
WARNING: The script kungfu-run is installed in '/home/user/.local/bin' which is not on PATH.
Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
```

Please add `kungfu-run` to your PATH:

```bash
PATH=$PATH:$HOME/.local/bin
```

**Important**: We provide a prepared VM that has already been set up with the above environment. To gain SSH access to such a VM, please contact the authors.

## Evaluation

We start by reproducing the performance benchmark results of KungFu. These benchmarks use a synthetic ImageNet workload and are easy to reproduce.

### 1. Monitoring Overhead (Figure 8)

In this experiment, we measure the overhead of computing online monitored training metrics: (i) gradient noise scale and (ii) gradient variance. To run this experiment, you need to start a VM that has 4 K80 GPUs. (In the paper, we present results from a DGX-1 machine with 8 V100 GPUs, but we can no longer provide dedicated access to this machine.)

You need to SSH to the VM. We first measure the training throughput of each GPU without monitoring:

```bash
kungfu-run -np 4 python3 benchmarks/monitoring/benchmark.py --kf-optimizer=sync-sgd --model=ResNet50 --batch-size=64
```

You should see an output with results as follows:

```text
...
[127.0.0.1.10002::stdout] Iter #0: 49.3 img/sec per /gpu:0
[127.0.0.1.10000::stdout] Iter #0: 49.4 img/sec per /gpu:0
[127.0.0.1.10001::stdout] Iter #0: 49.3 img/sec per /gpu:0
[127.0.0.1.10003::stdout] Iter #0: 49.3 img/sec per /gpu:0
...
```

To measure the overhead of computing **gradient noise scale**, we need to switch the `kf-optimizer` to `noise-scale`:

```bash
kungfu-run -np 4 python3 benchmarks/monitoring/benchmark.py --kf-optimizer=noise-scale --model=ResNet50 --batch-size=64 --interval==1
```

You should see the output below:

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

This shows that the training throughput drops from 49.2 to 48.7 images per second with extra gradient noise scale computation.

The same measurement is done for **gradient variance** monitoring:

```bash
kungfu-run -np 4 python3 benchmarks/monitoring/benchmark.py --kf-optimizer=variance --model=ResNet50 --batch-size=64 --interval==1
```

You should expect the following output:

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

### 2. Scalability (Figure 9)

In this experiment, we measure the scalability of the asynchronous collective communication layer in KungFu. We compare the ideal throughput (i.e., training throughput without
communication) and actual training throughput.

You need to launch a cluster that has `N` VMs. For example, with 2 VMs, we assume that their private IPs are `10.0.0.19` and `10.0.0.20` (bound to NIC `eth0`). You can launch the 2-VM data parallel training scenario by running the following command on **each** VM:

```bash
kungfu-run -np 2 -H 10.0.0.19:1,10.0.0.20:1 -nic=eth0 python3 benchmarks/system/benchmark_kungfu.py --kf-optimizer=sync-sgd --model=ResNet50 --batch-size=64
```

The `-H` parameter is in the format: `<ip1>:<slot>,<ip2>:<slot>` where `ip1` is usually the private IP and the `slot` is the number of GPUs per machine. The total number of GPUs is specified by the `-np` parameter.

You should observe the following ouptut on one of the VMs:

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

To run the scalability experiment using another model, `MobileNetV2`, you
need to replace `--model=ResNet50` with `--model=MobileNetV2`.

The same chgange can be applied to clusters with any number (i.e.,
8, 16, 32, ...) of VMs.

### 3. Scaling Performance (Figure 7)

[...]

### 4. Adaptive batch size (Figure 4)

In this experiment, we train the ResNet-56 model with the CIFAR10 dataset using fixed batch sizes (4 x 32 and 4 x 1024) and adaptive batch sizes. To run the experiment, you need a machine with 4 GPUs.

In addition to installing KungFu, you need to clone our fork of `http://github.com/tensorflow/models`.
You can use the following commands to clone the code and run the experiments:

```
git clone https://github.com/luomai/models -b osdi20
cd models

# Download the dataset to $HOME/var/data/cifar
./download-cifar10-data.sh

# Run the static baseline
./train-cifar10-fixed.sh

# Run the adaptive batch size
./train-cifar10-adaptive.sh

# Extract the TensorFlow logs and generate the plots; the results will be saved to ./data.
./plot-all.sh
```

### 5. Adaptive Communication Strategy (Figure 5)

In this experiment, we show how KungFu adapts trainig in the light of adversarial network conditions. We utilise low-level monitoring inside KungFu's communication stack to monitor the throughput from the all-reduce operations and detect network interference or contention. In such cases, the adaptation policy (AP) adjusts the communication strategy (i.e., the topology used by the all-reduce operations) to reduce the load on contended network links.

We simiulate network contention by manually introducing background traffic between the master node in the default communication strategy and an external node (i.e., a node that is not taking part in the training process).

For this experiment, we launch 4 VMs (but it can be any number above 2) with 1 K80 GPU per VM and 1 extra VM in the same VLAN for generating background traffic. We use the ResNet50 model for training and specify the communication strategy to be `STAR`.

To start the distributed training, we run the following command concurrently on all VMs that take part in the training process:

```bash
kungfu-run -np 4 -strategy STAR -H $HOSTS_VAR -nic eth0 python3 experimental/adapt_strategy/adapt_strategy.py --adapt --kf-optimizer=sync-sgd-monitor
```

where

```bash
HOSTS_VAR=<list of comma seperated IPs and processes per machine (e.g., 192.168.10.2:1, 192.168.10.2:2)>
```

We initiate the background traffic between the master node (note that the default master node is the first peer from the list defined in `HOSTS_VAR`) and the VM external to the training at an arbitrary time during training. We do so by invoking:

```bash
kungfu-run -np 32 -H $HOSTS_VAR -strategy STAR -nic eth0 -port-range 11100-11200 $HOME/go/bin/kungfu-bench-allreduce -model resnet50-imagenet -mode par -epochs 25
```

where

```bash
HOSTS_VAR=<masterNode>:1,<externalNode>:31
```

and `kungfu-bench-allreduce` is a network tool written in Go that creates synthetic background traffic. You will need to install it by invoking the following in the KungFu directory: 

```bash
go install ./...
```
You then find the executable in the `go/bin` directory.

After launchign both the training and the background traffic generation, you should see the following output:

```bash
[10.0.0.7.10000::stdout] Cluster response Iter #20: 30.2 img/sec per /gpu:0
[10.0.0.7.10000::stdout] [I] MonitorStrategy:: Checking Throughput = 119.70 MiB/s, reff= 92.11 MiB/s
[10.0.0.7.10000::stdout] [I] AP:: cluster response -> 0
[10.0.0.7.10000::stdout] Cluster response Iter #21: 29.4 img/sec per /gpu:0
[10.0.0.7.10000::stdout] [I] MonitorStrategy:: Checking Throughput = 114.74 MiB/s, reff= 92.11 MiB/s
[10.0.0.7.10000::stdout] [I] AP:: cluster response -> 0
[10.0.0.7.10000::stdout] Cluster response Iter #22: 26.5 img/sec per /gpu:0
[10.0.0.7.10000::stdout] [I] MonitorStrategy:: Checking Throughput = 101.00 MiB/s, reff= 92.11 MiB/s
[10.0.0.7.10000::stdout] [I] AP:: cluster response -> 1
[10.0.0.7.10000::stdout] Cluster response Iter #23: 29.4 img/sec per /gpu:0
[10.0.0.7.10000::stdout] [I] MonitorStrategy:: Checking Throughput = 114.95 MiB/s, reff= 92.11 MiB/s
[10.0.0.7.10000::stdout] [I] AP:: cluster response -> 0
[10.0.0.7.10000::stdout] Cluster response Iter #24: 31.4 img/sec per /gpu:0
[10.0.0.7.10000::stdout] [I] MonitorStrategy:: Checking Throughput = 124.96 MiB/s, reff= 92.11 MiB/s
[10.0.0.7.10000::stdout] [I] AP:: cluster response -> 0
[10.0.0.7.10000::stdout] Cluster response Iter #25: 28.9 img/sec per /gpu:0
[10.0.0.7.10000::stdout] [I] MonitorStrategy:: Checking Throughput = 112.70 MiB/s, reff= 92.11 MiB/s
[10.0.0.7.10000::stdout] [I] AP:: cluster response -> 0
[10.0.0.7.10000::stdout] Cluster response Iter #26: 28.7 img/sec per /gpu:0
[10.0.0.7.10000::stdout] [I] MonitorStrategy:: Checking Throughput = 111.98 MiB/s, reff= 92.11 MiB/s
[10.0.0.7.10000::stdout] [I] AP:: cluster response -> 0
[10.0.0.7.10000::stdout] Cluster response Iter #27: 28.0 img/sec per /gpu:0
[10.0.0.7.10000::stdout] [I] MonitorStrategy:: Checking Throughput = 108.53 MiB/s, reff= 92.11 MiB/s
[10.0.0.7.10000::stdout] [I] AP:: cluster response -> 0
[10.0.0.7.10000::stdout] Cluster response Iter #28: 25.8 img/sec per /gpu:0
[10.0.0.7.10000::stdout] [I] MonitorStrategy:: Checking Throughput = 97.76 MiB/s, reff= 92.11 MiB/s
[10.0.0.7.10000::stdout] [I] AP:: cluster response -> 0
[10.0.0.7.10000::stdout] Cluster response Iter #29: 27.9 img/sec per /gpu:0
[10.0.0.7.10000::stdout] [I] MonitorStrategy:: Checking Throughput = 108.60 MiB/s, reff= 92.11 MiB/s
[10.0.0.7.10000::stdout] [I] AP:: cluster response -> 0
[10.0.0.7.10000::stdout] Cluster response Iter #30: 28.6 img/sec per /gpu:0
[10.0.0.7.10000::stdout] [I] MonitorStrategy:: Checking Throughput = 111.05 MiB/s, reff= 92.11 MiB/s
[10.0.0.7.10000::stdout] [I] AP:: cluster response -> 0
[10.0.0.7.10000::stdout] Cluster response Iter #31: 19.2 img/sec per /gpu:0
[10.0.0.7.10000::stdout] [I] MonitorStrategy:: Checking Throughput = 68.82 MiB/s, reff= 92.11 MiB/s
[10.0.0.7.10000::stdout] [I] AP:: cluster response -> 7
[10.0.0.7.10000::stdout] [I] AP:: cluster reached consensus on changing to alternative strategy
[10.0.0.7.10000::stdout] Cluster response
[10.0.0.7.10000::stdout] Interference detected. Changing to alternative comm strategy !
[10.0.0.7.10000::stdout] Iter #32: 20.9 img/sec per /gpu:0
[10.0.0.7.10000::stdout] Iter #33: 23.3 img/sec per /gpu:0
[10.0.0.7.10000::stdout] Iter #34: 26.2 img/sec per /gpu:0
[10.0.0.7.10000::stdout] Iter #35: 26.9 img/sec per /gpu:0
```

At around iteration #29, we initiate the background traffic. The effects are visible from the next iterations when the measured throughput drops. After it has dropped below a predefined threshold in a reference window, KungFu initiates the change to an alternative communication strategy. Therefore we observe that, from iteration #32 onwards, the throughput recovers.

To measure the baseline execution scenario with no adaption enabled, you need to launch the training by running:

```bash
kungfu-run -q -np 4 -strategy STAR -H $HOSTS_VAR -nic eth0 python3 experimental/adapt_strategy/adapt_strategy.py --kf-optimizer=sync-sgd-monitor
```

### 6. Adaptive resource provisioning (Figure 6)

In the adaptive resource provisioning experiment, we use KungFu's support for elastic scaling to increase/decrease the cluster size and eventually select the optimal size. We measure the total training throughput and increase the number of workers with a fixed frequency. The addition of workers happens until the total througput has not increased more than a predefined threshold. In this case, the last added worker is removed, and the training is finished with that number of workers.

For this experiment, we use the BERT-base model and the Squad 2.0 dataset. We clone a fork of the BERT repository that has all the adjustments to work with KungFu. The adjustments for KungFu are done on the branch `kungfu-elastic-scaling`.
```bash
git clone https://github.com/marwage/bert.git
git checkout kungfu-elastic-scaling
```

The next step is to change to the `bert` directory.
```bash
cd bert
```

Since we do fine-tuning on the Bert-base model, we require a pretrained model.
The Zip file with the model can be downloaded from [Bert-base](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip).
After downloading, unzip the archive.

We also need to download the Squad 2.0 dataset.
The necessary files can be found here:
*   [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
*   [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
*   [evaluate-v2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

Place these files in a data directory.

Inside the `bert` directory, there is the shell script `run_elastic_scaling.sh`. Adjust the `BERT_BASE_DIR` and `SQUAD_DIR` variables to reflect the directories with the models.

To start the experiment, you need to run the following command:
```bash
./run_elastic_scaling.sh
```

The terminal output during the experimental run should look as follows:
```text
...
[127.0.0.1.10000::stderr] I0904 16:59:44.791337 139815124088640 tpu_estimator.py:2308] examples/sec: 12.6664
[127.0.0.1.10002::stderr] I0904 16:59:44.791208 140353563645760 tpu_estimator.py:2307] global_step/sec: 1.58332
[127.0.0.1.10002::stderr] INFO:tensorflow:examples/sec: 12.6666
[127.0.0.1.10002::stderr] I0904 16:59:44.792021 140353563645760 tpu_estimator.py:2308] examples/sec: 12.6666
[127.0.0.1.10001::stderr] INFO:tensorflow:global_step/sec: 1.58563
[127.0.0.1.10000::stderr] INFO:tensorflow:global_step/sec: 1.58544
[127.0.0.1.10000::stderr] I0904 16:59:45.421695 139815124088640 tpu_estimator.py:2307] global_step/sec: 1.58544
[127.0.0.1.10001::stderr] I0904 16:59:45.421675 139956496979776 tpu_estimator.py:2307] global_step/sec: 1.58563
[127.0.0.1.10002::stderr] INFO:tensorflow:global_step/sec: 1.58567
[127.0.0.1.10000::stderr] INFO:tensorflow:examples/sec: 12.6835
[127.0.0.1.10001::stderr] INFO:tensorflow:examples/sec: 12.6851
[127.0.0.1.10000::stderr] I0904 16:59:45.422121 139815124088640 tpu_estimator.py:2308] examples/sec: 12.6835
[127.0.0.1.10001::stderr] I0904 16:59:45.422118 139956496979776 tpu_estimator.py:2308] examples/sec: 12.6851
[127.0.0.1.10002::stderr] I0904 16:59:45.421859 140353563645760 tpu_estimator.py:2307] global_step/sec: 1.58567
...
```

After the experiment has finished, there are output files `out_{#}.csv` in the `bert` directory. In `out_{#}.csv`, we store, inter alia, the global step, number of workers, and the throughput of the workers. The file `out_0.csv` will have the complete history of the experiment because it is from the worker with rank 1. Move the `out_0.csv` file if you run more than one experiement because it will otherwise be overwritten. In the `tmp` directory, there are logs of KungFu and each worker.

To compare the optimal cluster size with running all workers from the start, you can use the shell script `run_all_workers.sh`. If you needed to adjust the `SQUAD_DIR` and `BERT_BASE_DIR` variables in the `run_elastic_scaling.sh` script, the same must be done in `run_all_workers.sh`.

The start command for this experiment is:
```bash
./run_all_workers.sh
```

The output during the experiment with all workers should look like the elastic scaling experiement.

To plot the results, we need to do the following:

For the python script `plot.py` to be able to run, those packages must be installed:
* pandas
* matplotlib
* numpy

Rename the output csv files to `no_scaling.csv` and `scaling.csv`.

```bash
python3 plot.py
```

To see the figure open `optimal_cluster_size.pdf`.
