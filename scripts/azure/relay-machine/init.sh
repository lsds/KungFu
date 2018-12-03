#!/bin/sh
set -e

if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -f .ssh/id_rsa -P ''
fi

cat .ssh/id_rsa.pub >>.ssh/authorized_keys

echo 'StrictHostKeyChecking=no' >>.ssh/config
