#!/bin/sh
set -e

cd $(dirname $0)
KUNGFU_ROOT=$PWD

stop_fuse() {
    local root=$1
    if [ $(mount | grep $root | wc -l) -gt 0 ]; then
        fusermount -u $root
    fi
}

remount() {
    local root=$1
    stop_fuse $root

    local idx_file=$2

    seed=0
    progress=10
    cluster_size=4
    global_batch_size=23
    max_sample_per_file=8192

    $KUNGFU_ROOT/bin/tfrecord-fs $idx_file $root $seed $progress $cluster_size $global_batch_size $max_sample_per_file
    mount | grep $root
}

main() {
    root=$HOME/mnt/tfrecords
    if [ ! -d root ]; then
        mkdir -p $root
    fi

    idx_file=$KUNGFU_ROOT/tf-index-1.idx.txt # SQuAD
    # idx_file=$KUNGFU_ROOT/tf-index-1024.idx.txt # ImageNet

    stop_fuse $root
    remount $root $idx_file

    # tree $root
}

main $@
$KUNGFU_ROOT/deps/tests/test_tf_record_fs.py
