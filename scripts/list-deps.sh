#!/bin/sh
set -e

cd $(dirname $0)/..
root=$(awk '{print $2}' go.mod | head -n 1)

godeps() {
    local m=$1
    local tab=$2
    for d in $(go list -deps $m | grep $root | grep -v $m); do
        echo "$tab$d"
    done
}

list_deps() {
    for m in $(go list ./srcs/...); do
        echo $m
        godeps $m "    "
    done
}

_tree_deps() {
    local m=$1
    local tab=$2
    echo "$tab$m"
    for d in $(go list -deps $m | grep $root | grep -v $m); do
        _tree_deps $d "    $tab"
    done
}

tree_deps() {
    for m in $(go list ./srcs/...); do
        _tree_deps $m ""
    done
}

list_deps
# tree_deps
