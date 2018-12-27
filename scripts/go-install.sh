#!/bin/sh
set -e

cd $(dirname $0)/..

GOBIN=$(pwd)/bin go install -v ./srcs/go/kungfu-run
