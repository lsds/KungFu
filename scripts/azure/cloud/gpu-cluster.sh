#!/bin/sh

set -e

cd $(dirname $0)/..

DEBUG=--debug

SUFFIX=test
IMAGE_GROUP=KungFu
IMAGE_NAME=cuda-9-cudnn-7
GROUP=KungFu-${SUFFIX}
LOCATION=eastus

VNET=default-vnet
NSG=default-nsg
SUBNET=default-subnet

ADMIN=kungfu
SIZE=Standard_NV24 # 4 GPU

IMAGE=$(az image show -g ${IMAGE_GROUP} -n ${IMAGE_NAME} --query id | tr -d '"')

measure() {
    local begin=$(date +%s)
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    echo "$@ took ${duration}s" | tee -a profile.log
}

create_vm() {
    local NAME=$1
    local IMAGE=$2
    local SIZE=$3

    az vm create \
        -g ${GROUP} \
        -n ${NAME} \
        --admin-user ${ADMIN} \
        --image ${IMAGE} \
        --size ${SIZE} \
        --vnet-name ${VNET} \
        --subnet ${SUBNET} \
        --nsg ${NSG} \
        -o table ${DEBUG}
}

create_node() {
    create_vm $1 $IMAGE $SIZE
}

create_relay() {
    local NAME=relay
    local IMAGE=Canonical:UbuntuServer:18.04-LTS:latest
    local SIZE=Standard_DS3_v2
    create_vm $NAME $IMAGE $SIZE
}

setup_group() {
    if [ $(az group exists -n ${GROUP}) == "false" ]; then
        az group create -l ${LOCATION} -n ${GROUP} -o table ${DEBUG}
    fi
}

setup_network() {
    az network vnet create -g ${GROUP} -n ${VNET} --subnet-name ${SUBNET} -o table ${DEBUG}
    az network nsg create -g ${GROUP} -n ${NSG} -o table ${DEBUG}
    az network nsg rule create -g ${GROUP} --nsg-name ${NSG} \
        -n allow-ssh --priority 1000 --destination-port-ranges 22 -o table ${DEBUG}
}

setup_nodes() {
    create_relay &
    create_node node-01 &
    create_node node-02 &
    wait
}

get_ip() {
    local NAME=$1
    az vm list-ip-addresses -g ${GROUP} -n ${NAME} --query '[0].virtualMachine.network.publicIpAddresses[0].ipAddress' | tr -d '"'
}

get_internal_ip() {
    local NAME=$1
    az vm list-ip-addresses -g ${GROUP} -n ${NAME} --query '[0].virtualMachine.network.privateIpAddresses[0]' | tr -d '"'
}

send_authorized_keys() {
    local RELAY_IP=$1
    local NODE_IP=$(get_ip $2)
    local NODE_INTERNAL_IP=$(get_internal_ip $2)

    echo "$2 ${NODE_INTERNAL_IP}" >>nodes.txt
    scp authorized_keys ${ADMIN}@${NODE_IP}:~/.ssh/authorized_keys
}

setup_keys() {
    local RELAY_IP=$(get_ip relay)

    scp -r relay-machine $ADMIN@${RELAY_IP}:~/
    ssh $ADMIN@${RELAY_IP} ./relay-machine/init.sh

    scp ${ADMIN}@${RELAY_IP}:~/.ssh/authorized_keys .

    [ -f nodes.txt ] && rm nodes.txt
    send_authorized_keys $RELAY_IP node-01 &
    send_authorized_keys $RELAY_IP node-02 &
    wait

    scp nodes.txt $ADMIN@${RELAY_IP}:~/
    ssh $ADMIN@${RELAY_IP} ./relay-machine/test-nodes.sh
}

create_cluster() {
    measure setup_group
    measure setup_network
    measure setup_nodes
    measure setup_keys
}

delete_cluster() {
    measure az group delete -n ${GROUP} --yes --debug
}

main() {

    if [ "$1" == "ssh" ]; then
        ssh ${ADMIN}@$(get_ip relay)
    elif [ "$1" == "down" ]; then
        measure delete_cluster
    else
        # create_cluster took 270s
        measure create_cluster
    fi
}

main $@
