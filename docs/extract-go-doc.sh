#!/bin/sh
set -e

cd $(dirname $0)/..

gen_doc() {
    for p in $(go list ./srcs/...); do
        go run docs/extract.go -pkg $p
    done
}

gen_doc | tee docs/go.md
