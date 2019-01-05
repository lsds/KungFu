#!/bin/sh
set -e

if [ ! -f $HOME/.ssh/id_rsa ]; then
    ssh-keygen -f $HOME/.ssh/id_rsa -P ''
fi

PUB_KEY=$(cat ~/.ssh/id_rsa.pub)

if ! grep -q "${PUB_KEY}" $HOME/.ssh/authorized_keys; then
    echo ${PUB_KEY} >>$HOME/.ssh/authorized_keys
fi

touch $HOME/.ssh/config
if ! grep -q 'StrictHostKeyChecking=no' $HOME/.ssh/config; then
    echo 'StrictHostKeyChecking=no' >>$HOME/.ssh/config
fi
