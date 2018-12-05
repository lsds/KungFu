#!/bin/sh
set -e

cd $HOME

# ansible-playbook -i hosts.txt relay-machine/playbook.yaml --list-hosts
ansible-playbook -i hosts.txt -v relay-machine/playbook.yaml
