#!/bin/sh
set -e
cd $(dirname $0)/..

PKG=$(cat go.mod | awk '{print $2}')

show_internal_deps() {
    local full_name=$(go list $1)
    echo "$full_name depends on:"
    for d in $(go list -f '{{ join .Deps "\n" }}' $full_name | grep $PKG); do
        echo "- $d"
    done
    echo
}

show_internal_deps ./srcs/go/plan
show_internal_deps ./srcs/go/rchannel
show_internal_deps ./srcs/go/kungfubase
show_internal_deps ./srcs/go/kungfuconfig
show_internal_deps ./srcs/go/kungfu
show_internal_deps ./srcs/go/libkungfu-comm
