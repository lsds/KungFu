#!/bin/sh
set -e

cd $(dirname $0)

measure() {
    local begin=$(date +%s)
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    echo "$@ took ${duration}s" | tee -a $HOME/init.log
}

download_data() {
    env \
        DATA_DIR=$HOME/var/data \
        DATA_MIRROR_PREFIX=https://kungfudata.blob.core.windows.net/data \
        ./download-mnist.sh
}

main() {
    sudo sed -i 's/APT::Periodic::Update-Package-Lists "1";/APT::Periodic::Update-Package-Lists "0";/' /etc/apt/apt.conf.d/10periodic
    cat /etc/apt/apt.conf.d/10periodic

    measure ./install-cuda-9.sh
    measure ./install-cudnn-7.sh
    measure ./install-tensorflow-gpu.sh

    measure sudo apt install -y build-essential cmake iperf nload htop
    measure ./install-golang1.11.sh
    measure ./install-openmpi.sh

    measure download_data

    cp .my_bashrc $HOME

    if ! grep '.my_bashrc' /home/vagrant/.bashrc; then
        echo '[ -f ~/.my_bashrc ] && . ~/.my_bashrc' >>$HOME/.bashrc
    fi
}

measure main
