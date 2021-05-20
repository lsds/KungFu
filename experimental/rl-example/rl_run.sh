#!/bin/sh
set -e

cd $(dirname $0)

. ../../scripts/launcher.sh

n_leaner=2
n_actor=2
n_server=1

flags() {
    echo -l $n_leaner
    echo -a $n_actor
    echo -s $n_server
}

rl_run() {
    local n=$((n_leaner + n_actor + n_server))
    prun $n python3 rl_agent.py $(flags)
}

main() {
    rl_run
}

main
