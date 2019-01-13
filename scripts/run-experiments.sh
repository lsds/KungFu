#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_NAME=$(basename $0)
SCRIPT_DIR=$(pwd)

. ./utils/measure.sh

cd ../..

main() {
    measure $SCRIPT_DIR/azure/cloud/upload-to-relay.sh

    measure $SCRIPT_DIR/azure/cloud/relay-run.sh \
        ./relay-machine/run-experiments.sh prepare

    measure $SCRIPT_DIR/azure/cloud/relay-run.sh \
        ./relay-machine/run-experiments.sh run

    measure $SCRIPT_DIR/azure/cloud/download-report.sh
}

measure main
