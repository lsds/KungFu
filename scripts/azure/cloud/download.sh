#!/bin/sh

# download logs from relay
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

main() {
    local RELAY_IP=$(get_ip relay)
    measure scp -r $ADMIN@$RELAY_IP:~/logs .

    cd logs
    find . -name *.tar.bz2 -exec tar -xf {} \;
}

measure main
