#!/bin/sh

set -e
set -x

SCRIPT_NAME=$(basename $0)
cd $(dirname $0)/..
. ../utils/show_duration.sh

# DEBUG=--debug

if [ -z "${PREFIX}" ]; then
    PREFIX=$USER-test-cluster
fi
IMAGE_GROUP=KungFu
GROUP=KungFu
LOCATION=eastus

IMAGE_NAME=cuda-9-cudnn-7
VNET=${PREFIX}-vnet
NSG=${PREFIX}-nsg
SUBNET=${PREFIX}-subnet

ADMIN=kungfu
SIZE=Standard_NV24 # 4 GPU

RELAY_NAME=${PREFIX}-relay
RELAY_IMAGE_NAME=relay-ubunbu18
# RELAY_IMAGE=Canonical:UbuntuServer:18.04-LTS:latest
RELAY_SIZE=Standard_DS3_v2

if [ -z "${N_NODES}" ]; then
    N_NODES=1
fi

IMAGE=$(az image show -g ${IMAGE_GROUP} -n ${IMAGE_NAME} --query id | tr -d '"')
RELAY_IMAGE=$(az image show -g ${IMAGE_GROUP} -n ${RELAY_IMAGE_NAME} --query id | tr -d '"')

node_names() {
    for i in $(seq $1); do
        printf "${PREFIX}-node-%02d\n" $i
    done
}

ALL_NODES=$(node_names ${N_NODES})

measure() {
    local begin=$(date +%s)
    echo "[begin] $SCRIPT_NAME::$@ at $begin"
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    local dur=$(show_duration $duration)
    echo "[done] $SCRIPT_NAME::$@ took ${dur}" | tee -a profile.log
}

delete_resource() {
    local NAME=$1
    local TYPE=$2
    az resource delete -g ${GROUP} --resource-type ${TYPE} -n ${NAME}
}

delete_nic() {
    delete_resource "$1" Microsoft.Network/networkInterfaces
}

delete_ip() {
    delete_resource "$1" Microsoft.Network/publicIPAddresses
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

delete_vm() {
    local NAME=$1
    az vm delete -g ${GROUP} -n ${NAME} --yes ${DEBUG}
}

create_node() {
    create_vm $1 $IMAGE $SIZE
}

create_relay() {
    create_vm $RELAY_NAME $RELAY_IMAGE $RELAY_SIZE
}

setup_group() {
    if [ $(az group exists -n ${GROUP}) = "false" ]; then
        # az group create -l ${LOCATION} -n ${GROUP} -o table ${DEBUG}
        exit 1
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
    for node in ${ALL_NODES}; do
        create_node ${node} &
    done
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
    local RELAY_IP=$(get_ip ${RELAY_NAME})

    scp -r relay-machine $ADMIN@${RELAY_IP}:~/
    ssh $ADMIN@${RELAY_IP} ./relay-machine/user-init.sh

    scp ${ADMIN}@${RELAY_IP}:~/.ssh/authorized_keys .

    [ -f nodes.txt ] && rm nodes.txt
    for node in ${ALL_NODES}; do
        send_authorized_keys $RELAY_IP ${node} &
    done
    wait

    scp nodes.txt $ADMIN@${RELAY_IP}:~/
    ssh $ADMIN@${RELAY_IP} ./relay-machine/test-nodes.sh
}

create_cluster() {
    # create_cluster takes about 270s
    measure setup_group
    measure setup_network
    measure setup_nodes
    measure setup_keys
}

delete_net_for() {
    local NAME=$1
    delete_nic ${NAME}VMNic
    delete_ip ${NAME}PublicIP

}

delete_cluster() {
    delete_vm ${RELAY_NAME} &
    for node in ${ALL_NODES}; do
        delete_vm ${node} &
    done
    wait

    delete_net_for ${RELAY_NAME} &
    for node in ${ALL_NODES}; do
        delete_net_for ${node} &
    done
    wait

    delete_resource ${NSG} Microsoft.Network/networkSecurityGroups
    delete_resource ${VNET} Microsoft.Network/virtualNetworks &
    for disk in $(az resource list -g ${GROUP} --resource-type Microsoft.Compute/disks --query '[].id' -o table | grep ${PREFIX}); do
        az resource delete --id $disk &
    done
    wait
}

reload_cluster() {
    measure delete_cluster
    measure create_cluster
}

main() {
    if [ "$1" = "ssh" ]; then
        ssh ${ADMIN}@$(get_ip relay)
    elif [ "$1" = "up" ]; then
        measure create_cluster
    elif [ "$1" = "down" ]; then
        measure delete_cluster
    elif [ "$1" = "reload" ]; then
        measure reload_cluster
    else
        measure create_cluster
    fi
}

measure main $@
