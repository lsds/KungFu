#!/bin/sh
set -e

cd $(dirname $0)/..
root=$(awk '{print $2}' go.mod | head -n 1)

for m in $(go list ./srcs/...); do
    echo $m
    for d in $(go list -deps $m | grep $root | grep -v $m); do
        echo "    $d"
    done
done
