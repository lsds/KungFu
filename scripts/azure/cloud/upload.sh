#!/bin/sh

# upload repo to relay
set -e

SUFFIX=test
GROUP=KungFu-${SUFFIX}
ADMIN=kungfu

measure() {
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

cd $(dirname $0)/../../../..

# VERBOSE=-v

main() {
    local RELAY_IP=$(get_ip relay)

    [ -f kungfu.tar ] && rm kungfu.tar
    [ -f kungfu.tar.bz2 ] && rm kungfu.tar.bz2
    tar -cf kungfu.tar kungfu
    bzip2 kungfu.tar
    du -hs kungfu.tar.bz2

    measure scp ${VERBOSE} kungfu.tar.bz2 $ADMIN@$RELAY_IP:~/
    measure ssh ${VERBOSE} $ADMIN@$RELAY_IP \
        sh -c '"rm -fr kungfu && tar -xf kungfu.tar.bz2 && cp -vr kungfu/scripts/azure/relay-machine . "'

    measure ssh ${VERBOSE} $ADMIN@$RELAY_IP ./relay-machine/ansible.sh
    measure ssh ${VERBOSE} $ADMIN@$RELAY_IP ./relay-machine/play.sh
}

measure main
