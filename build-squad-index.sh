#!/bin/sh
set -e

cd $(dirname $0)

list_squad_records() {
    echo /data/squad1/train.tf_record
}

export STD_TRACER_PATIENT=1

MS_BUILD_TF_INDEX=$PWD/bin/ms-elastic-build-tf-index

$MS_BUILD_TF_INDEX $(list_squad_records)
