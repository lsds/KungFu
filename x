#!/bin/sh

set -e

build() {
    GOBIN=$PWD/bin go install -v ./srcs/go/cmd/kungfu-elastic-run
    GOBIN=$PWD/bin go install -v ./tests/go/cmd/kungfu-test-elastic-worker
}

elastic_run_n_flags() {
    local np=$1

    echo -q
    echo -logdir logs

    echo -w

    echo -np $np
}

elastic_run_n() {
    local np=$1
    shift

    $PWD/bin/kungfu-elastic-run $(elastic_run_n_flags $np) $@
}

app_flags() {
    echo -idx-file $PWD/../../tf-index-1.idx.txt
    echo -max-progress 88641

    # echo -batch-size 24
    echo -batch-size $((1 << 12))
}

main() {
    elastic_run_n 1 $PWD/bin/kungfu-test-elastic-worker $(app_flags)
}

./INSTALL
build
./deps/build.sh
./deps/run.sh
# main
