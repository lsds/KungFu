#!/bin/sh
set -e

reinstall() {
    pip3 install -U .
}

# reinstall
./experimental/rl-example/run.sh
