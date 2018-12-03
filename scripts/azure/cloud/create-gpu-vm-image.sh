# This script is used for building a GPU VM image.
# You are more likely want to used the pre-build image rather than build a new one.

set -e

cd $(dirname $0)/..

GROUP=KungFu-tmp
LOCATION=eastus
NAME=gpu-template
ADMIN=kungfu

OUTPUT_IMAGE=cuda-9-cudnn-7
OUTPUT_GROUP=KungFu

# OS Image
IMAGE=Canonical:UbuntuServer:16.04-LTS:latest

# Machine type
SIZE=Standard_NV6 # 1 GPU

measure() {
    local begin=$(date +%s)
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    echo "$@ took ${duration}s" | tee -a profile.log
}

create_vm() {
    az group create -l ${LOCATION} -n ${GROUP} -o table --debug
    az vm create -g ${GROUP} -n ${NAME} --admin-user ${ADMIN} --image ${IMAGE} --size ${SIZE} -o table --debug
}

down() {
    az group delete -n ${GROUP} --yes --debug
}

get_ip() {
    az vm list-ip-addresses -g ${GROUP} -n ${NAME} --query '[0].virtualMachine.network.publicIpAddresses[0].ipAddress' | tr -d '"'
}

send_scripts() {
    scp -r ./gpu-machine ${ADMIN}@$(get_ip):~/
}

install_vm() {
    local ip=$(get_ip)
    measure ssh ${ADMIN}@$ip ./gpu-machine/init.sh
    measure az vm restart -g ${GROUP} -n ${NAME} --debug

    sleep 60 # TODO: wait until the VM can be ssh into
    measure ssh ${ADMIN}@$ip ./gpu-machine/test-tf-gpu.py
}

save_vm() {
    # https://docs.microsoft.com/en-us/azure/virtual-machines/linux/capture-image
    az vm deallocate -g ${GROUP} -n ${NAME} --debug
    az vm generalize -g ${GROUP} -n ${NAME} --debug

    local VM_ID=$(az resource show -g ${GROUP} -n ${NAME} --resource-type Microsoft.Compute/virtualMachines --query 'id' --debug | tr -d '"')
    measure az image create -g ${OUTPUT_GROUP} -n ${OUTPUT_IMAGE} --source ${VM_ID} -o table --debug
}

main() {
    measure create_vm
    measure send_scripts
    measure install_vm
    measure save_vm
    measure down
}

measure main $@
