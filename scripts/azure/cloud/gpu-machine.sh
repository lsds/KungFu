#!/bin/sh

set -e

cd $(dirname $0)/..

GROUP=KungFu
LOCATION=eastus
# LOCATION=southeastasia
NAME=gpu-dev
ADMIN=kungfu

# OS Image
# IMAGE=UbuntuLTS
IMAGE=Canonical:UbuntuServer:16.04-LTS:latest

# Machine type
# SIZE=Standard_NV6 # 1 GPU
# SIZE=Standard_NV12 # 2 GPU
SIZE=Standard_NV24 # 4 GPU

# SIZE=Standard_DS1_v2
# SIZE=Standard_D4s_v3 # 4 cores
# SIZE=Standard_D8s_v3 # 8 cores
# SIZE=Standard_D16s_v3 # 16 cores
# SIZE=Standard_D32s_v3 # 32 cores
# SIZE=Standard_D64s_v3 # 64 cores

measure() {
    local begin=$(date +%s)
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    echo "$@ took ${duration}s"
}

up() {
    # az group create -l southeastasia -n ${GROUP} -o table --debug
    az vm create -g ${GROUP} -l ${LOCATION} -n ${NAME} --image ${IMAGE} --size ${SIZE} --admin-user ${ADMIN} -o table --debug
    # az vm extension set --publisher Microsoft.Azure.Extensions --name DockerExtension -g ${GROUP} --vm-name ${NAME} -o table --debug
    az vm list-ip-addresses -g ${GROUP} -o table
    az vm open-port --port 8080 -g ${GROUP} -n ${NAME} --debug -o table
}

start() {
    az vm start -g ${GROUP} -n ${NAME} --debug
}

stop() {
    az vm stop -g ${GROUP} -n ${NAME} --debug
}

down() {
    az vm delete -g ${GROUP} -n ${NAME} --yes --debug
}

get_ip() {
    az vm list-ip-addresses -g ${GROUP} -n ${NAME} --query '[0].virtualMachine.network.publicIpAddresses[0].ipAddress' | tr -d '"'
}

send_scripts() {
    local IP=$(get_ip)
    echo $IP
    scp -r gpu-machine ${ADMIN}@$(get_ip):~/
    scp -r experiments ${ADMIN}@$(get_ip):~/
}

main() {
    if [ "$1" == "ssh" ]; then
        ssh ${ADMIN}@$(get_ip)
    elif [ "$1" == "down" ]; then
        measure down
    elif [ "$1" == "stop" ]; then
        measure stop
    elif [ "$1" == "start" ]; then
        measure start
    elif [ "$1" == "upload" ]; then
        measure send_scripts
    else
        measure up
        send_scripts
    fi
}

main $@
