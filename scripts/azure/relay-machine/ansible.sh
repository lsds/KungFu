#!/bin/sh
set -e

cd $HOME

ansible -i hosts.txt all -m ping

# ansible -i hosts.txt all -m shell -a 'rm -fr kungfu && rm kungfu.tar.bz2'

echo "distributing package"
# https://docs.ansible.com/ansible/latest/modules/unarchive_module.html
ansible -i hosts.txt all -m unarchive -a 'src=kungfu.tar.bz2 dest=.'

echo "downloading test data"
# https://docs.ansible.com/ansible/latest/modules/shell_module.html
ansible -i hosts.txt all -m shell -a './kungfu/scripts/download-mnist.sh'

echo "installing"
ansible -i hosts.txt all -m shell -a './kungfu/scripts/install.sh'
