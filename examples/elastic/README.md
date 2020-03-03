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
kungfu-cluster-manager-example -ttl 1m
```
