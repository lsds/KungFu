#!/bin/sh
set -e

cd $(dirname $0)
. ../../scripts/launcher.sh

app_flags() {
    true
}

main() {
    # prun 4 ./bin/queue-example $(app_flags)
    prun 4 python3 queue_example.py $(app_flags)
}

GOBIN=$PWD/bin go install -v ./...
main
