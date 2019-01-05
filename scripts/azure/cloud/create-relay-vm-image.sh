# This script is used for building a relay VM image.
# You are more likely want to used the pre-build image rather than build a new one.

set -e
set -x

cd $(dirname $0)/..

GROUP=KungFu
LOCATION=eastus
NAME=relay-template
ADMIN=kungfu

OUTPUT_IMAGE=relay-ubunbu18
OUTPUT_GROUP=KungFu

# OS Image
IMAGE=Canonical:UbuntuServer:18.04-LTS:latest

# Machine type
SIZE=Standard_DS3_v2

measure() {
    local begin=$(date +%s)
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    echo "$@ took ${duration}s" | tee -a profile.log
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
    az vm create -g ${GROUP} -n ${NAME} --admin-user ${ADMIN} --image ${IMAGE} --size ${SIZE} -o table --debug
}

delete_vm() {
    az vm delete -g ${GROUP} -n ${NAME} --yes --debug

    delete_nic ${NAME}VMNic
    delete_ip ${NAME}PublicIP
    delete_resource ${NAME}NSG Microsoft.Network/networkSecurityGroups
    delete_resource ${NAME}VNET Microsoft.Network/virtualNetworks

    for disk in $(az resource list -g ${GROUP} --resource-type Microsoft.Compute/disks --query '[].id' -o table | grep ${NAME}); do
        az resource delete --id $disk
    done
}

get_ip() {
    az vm list-ip-addresses -g ${GROUP} -n ${NAME} --query '[0].virtualMachine.network.publicIpAddresses[0].ipAddress' | tr -d '"'
}

send_scripts() {
    scp -r ./relay-machine ${ADMIN}@$(get_ip):~/
}

install_vm() {
    local ip=$(get_ip)
    measure ssh ${ADMIN}@$ip ./relay-machine/init.sh
    # measure az vm restart -g ${GROUP} -n ${NAME} --debug
    # sleep 60 # TODO: wait until the VM can be ssh into
    # measure ssh ${ADMIN}@$ip pwd
}

save_vm() {
    # https://docs.microsoft.com/en-us/azure/virtual-machines/linux/capture-image
    az vm deallocate -g ${GROUP} -n ${NAME} --debug
    az vm generalize -g ${GROUP} -n ${NAME} --debug

    local VM_ID=$(az resource show -g ${GROUP} -n ${NAME} --resource-type Microsoft.Compute/virtualMachines --query 'id' --debug | tr -d '"')

    measure az image delete -g ${OUTPUT_GROUP} -n ${OUTPUT_IMAGE} --debug
    measure az image create -g ${OUTPUT_GROUP} -n ${OUTPUT_IMAGE} --source ${VM_ID} -o table --debug
}

main() {
    measure create_vm
    measure send_scripts
    measure install_vm
    measure save_vm
    measure delete_vm
}

measure main $@
