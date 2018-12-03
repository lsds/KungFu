#!/bin/sh
set -e

for node in $(awk '{print $1}' ~/nodes.txt); do
    echo "testing $node"
    ssh $node pwd
    echo "$node is accessible from relay"
done
