#!/bin/sh

# upload repo to relay
set -e

if [ -z "${PREFIX}" ]; then
    PREFIX=$USER-test-cluster
fi

GROUP=KungFu
ADMIN=kungfu
RELAY_NAME=${PREFIX}-relay

measure() {
    echo "begin $@"
    local begin=$(date +%s)
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    echo "$@ took ${duration}s"
}

get_ip() {
    local NAME=$1
    az vm list-ip-addresses -g ${GROUP} -n ${NAME} --query '[0].virtualMachine.network.publicIpAddresses[0].ipAddress' | tr -d '"'
}

main() {
    # VERBOSE=-v
    local RELAY_IP=$(get_ip ${RELAY_NAME})
    echo "using RELAY_IP=$RELAY_IP"

    [ -f kungfu.tar ] && rm kungfu.tar
    [ -f kungfu.tar.bz2 ] && rm kungfu.tar.bz2
    tar -cf kungfu.tar kungfu
    bzip2 kungfu.tar
    du -hs kungfu.tar.bz2

    measure scp ${VERBOSE} kungfu.tar.bz2 $ADMIN@$RELAY_IP:~/
    measure ssh ${VERBOSE} $ADMIN@$RELAY_IP sh -c '"rm -fr kungfu && tar -xf kungfu.tar.bz2"'
}

measure main
