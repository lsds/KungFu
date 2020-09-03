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

The evaluation environment is hosted by a public cloud platform: Microsoft Azure. The base Virtual Machine (VM) image is [...] and you need to install
the following drivers and packages:

[...]

Once the VM is ready, you would need to install the KungFu library as follow:

[...]

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
