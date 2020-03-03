elastic training example

## How to run example

### build docker image
```bash
./.github/workflows/build-image.sh
```


### build cluster manager example

```bash
go install -v ./tests/go/cmd/kungfu-cluster-manager-example
```

### run examples

* simple example without tensorflow

```bash
kungfu-cluster-manager-example -ttl 1m kungfu-fake-adaptive-trainer
```

* train mnist SLP with tensorflow

```bash
kungfu-cluster-manager-example -ttl 1m \
    python3 ./examples/elastic/mnist_slp_estimator.py \
    --data-dir /root/var/data/mnist
```
