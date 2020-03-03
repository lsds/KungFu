# Elastic Training

This is an alpha example that demonstrate how to run elastic training with KungFu.

A dynamic KungFu cluster has three elements:

* An elastic TensorFlow training program should contain an `ElasticHook` provided by KungFu. This hook is responsible for tracking the changes of cluster membership and enforces changes without breaking the current training process.
* All KungFu workers (which runs the TensorFlow training program) synchronize the full cluster membership through a `ConfigStore`. To change the cluster membership, you can use `ConfigStoreClient` to write the new configuration into the `ConfigStore`.
* You should develop your own cluster manager to start and reclaim nodes during training. Once you start the nodes properly, you can write the new configuration into the `ConfigStore`.

In the following, we provide an example that go through the basic process of elastic training.

## Docker Image

In the cluster, we assume each server is initialized by Docker.

```bash
./.github/workflows/build-image.sh
```

## Cluster Manager Example

We provide an example that show how does a cluster manager work.
In this example, you can find how to use the `ConfigStoreClient` to update `ConfigStore`.

You can install the cluster manager example as follow:

```bash
go install -v ./tests/go/cmd/kungfu-cluster-manager-example
```

## Run Examples

We provide two options to bootstrap the cluster manager example.

### Training without TensorFlow

A simple training example without TensorFlow (testing purpose):

```bash
kungfu-cluster-manager-example -ttl 1m kungfu-fake-adaptive-trainer
```

### Training with TensorFLow

A full training example using TensorFlow:

```bash
kungfu-cluster-manager-example -ttl 1m \
    python3 ./examples/elastic/mnist_slp_estimator.py \
    --data-dir /root/var/data/mnist \
    --num-epochs 5
```

### Caveats

* full update (all old peers are removed in the new cluster) is not supported
* rank 0 of the new cluster must be a member of the old cluster
