elastic training example

## How to run example

### build image
```bash
./.github/workflows/build-image.sh
```


### build tools

```bash
go install -v ./tests/go/cmd/...
```

### run example

```bash
kungfu-cluster-manager-example -ttl 1m kungfu-fake-adaptive-trainer
```

```bash
kungfu-cluster-manager-example -ttl 1m \
    python3 ./examples/elastic/mnist_slp_estimator.py \
    --data-dir /root/var/data/mnist
```
