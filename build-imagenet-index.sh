#!/bin/sh
set -e

cd $(dirname $0)

list_tf_records() {
    ls /data/imagenet/records/train* | sort
}

export STD_TRACER_PATIENT=1

MS_BUILD_TF_INDEX=$PWD/bin/ms-elastic-build-tf-index

# 1281167
# 1281152, 15 dropped

$MS_BUILD_TF_INDEX $(list_tf_records)
