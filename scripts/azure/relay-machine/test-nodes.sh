#!/bin/sh
set -e

for node in $(awk '{print $1}' ~/nodes.txt); do
    echo "testing $node"
    ssh $node pwd
    echo "$node is accessible from relay"
done

awk '{print $2}' ~/nodes.txt >~/hosts.txt
ansible -i hosts.txt all -m shell -a pwd
