#!/bin/sh
set -e

SCRIPT_NAME=$(basename $0)

if [ -z "${PREFIX}" ]; then
    PREFIX=$USER-test-cluster
fi

GROUP=KungFu
ADMIN=kungfu
RELAY_NAME=${PREFIX}-relay

measure() {
    local begin=$(date +%s)
    echo "[begin] $SCRIPT_NAME::$@ at $begin"
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    echo "[done] $SCRIPT_NAME::$@ took ${duration}s"
}

get_ip() {
    local NAME=$1
    az vm list-ip-addresses -g ${GROUP} -n ${NAME} --query '[0].virtualMachine.network.publicIpAddresses[0].ipAddress' | tr -d '"'
}

main() {
    # VERBOSE=-v
    local RELAY_IP=$(get_ip ${RELAY_NAME})
    echo "using RELAY_IP=$RELAY_IP"
    measure scp ${VERBOSE} $ADMIN@$RELAY_IP:~/kungfu/experiment-results.txt .
}

measure main
