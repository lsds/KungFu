#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)

measure() {
    local begin=$(date +%s)
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    echo "$@ took ${duration}s" | tee -a $HOME/init.log
}

main() {
    sudo sed -i 's/APT::Periodic::Update-Package-Lists "1";/APT::Periodic::Update-Package-Lists "0";/' /etc/apt/apt.conf.d/10periodic
    cat /etc/apt/apt.conf.d/10periodic

    measure sudo apt update
    measure sudo apt install -y ansible tree python3 python3-pip build-essential cmake

    sudo cp -v -r $SCRIPT_DIR/ansible /usr/share/

    measure $SCRIPT_DIR/install-golang1.11.sh

    export LC_ALL=C
    measure pip3 install tensorflow
    measure ./install-openmpi.sh
}

measure main
