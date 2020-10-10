# Elastic Training

This is an alpha example that demonstrate how to run elastic training with KungFu.

A dynamic KungFu cluster has three elements:

* An elastic TensorFlow training program should contain an `ElasticHook` provided by KungFu. This hook is responsible for tracking the changes of cluster membership and enforces changes without breaking the current training process.
* All KungFu workers (which runs the TensorFlow training program) synchronize the full cluster membership through a `configServer`. To change the cluster membership, you can use [configClient](https://github.com/lsds/KungFu/blob/master/tests/go/cmd/kungfu-cluster-manager-example/configclient.go) to write the new configuration into the `configServer`.
* You should develop your own cluster manager to start and reclaim nodes during training. Once you start the nodes properly, you can write the new configuration into the `configServer`.

## Current Limitations

* Full update (all old peers are removed in the new cluster) is not supported.
* The Rank-0 worker in the new cluster must be part of the old cluster.

## Example

We provide an example that illustrate a typical elastic training program.

### Docker Image

In the cluster, we assume each server is initialized by Docker.

```bash
./.github/workflows/build-image.sh
```

### Cluster Manager Example

We provide an example that show how does a cluster manager work.
In this example, you can find how to use the `configClient` to update `configServer`.

You can install the cluster manager example as follow:

```bash
go install -v ./tests/go/cmd/kungfu-cluster-manager-example
```

### Run Examples

We provide two options to bootstrap the cluster manager example.

* A simple training example without TensorFlow (testing purpose):

```bash
kungfu-cluster-manager-example kungfu-fake-adaptive-trainer
```

* A full training example using TensorFlow:

```bash
kungfu-cluster-manager-example \
    python3 ./examples/mnist_elastic_docker/mnist_slp_estimator.py \
    --data-dir /root/var/data/mnist \
    --num-epochs 5
```
