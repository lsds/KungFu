#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_NAME=$(basename $0)
SCRIPT_DIR=$(pwd)

. ./utils/show_duration.sh

cd ../..

measure() {
    local begin=$(date +%s)
    echo "[begin] $SCRIPT_NAME::$@ at $begin"
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    local dur=$(show_duration $duration)
    echo "[done] $SCRIPT_NAME::$@ took ${dur}" | tee -a $SCRIPT_DIR/profile.log
}

main() {
    measure $SCRIPT_DIR/azure/cloud/upload-to-relay.sh

    measure $SCRIPT_DIR/azure/cloud/relay-run.sh \
        ./relay-machine/run-experiments.sh prepare

    measure $SCRIPT_DIR/azure/cloud/relay-run.sh \
        ./relay-machine/run-experiments.sh run

    measure $SCRIPT_DIR/azure/cloud/download-report.sh
}

measure main
