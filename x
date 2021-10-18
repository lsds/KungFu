#!/bin/sh

set -e

cd $(dirname $0)
KF_ROOT=$PWD
MS_ROOT=$HOME/code/repos/github.com/kungfu-ml/kungfu-mindspore/mindspore

echo "MS_ROOT: ${MS_ROOT}"

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

# ./INSTALL
# build
# ./deps/build.sh

# rm -fr *.tf_record
# rm -fr *.list.txt
# ./deps/run.sh
# # main

./deps/build.sh
./deps/mlfs/build.sh
./remount-tf-records.sh

# python3 deps/tests/tf_read_tf_records.py # doesn't work

. $MS_ROOT/../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path $MS_ROOT)
echo $LD_LIBRARY_PATH | tr ':' '\n'

PYTHON=$(which python3.8)
# $PYTHON deps/tests/ms_read_tf_records.py
