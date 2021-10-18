#!/bin/sh

# set -e

root=$HOME/mnt/tfrecords

stop_fuse() {
    local root=$1
    if [ $(mount | grep $root | wc -l) -gt 0 ]; then
        fusermount -u $root
    fi
}

stop_fuse $root

mount | grep fuse

echo "done $0"
