#!/bin/sh
set -e

reinstall() {
    rm -fr setup.py
    ln -s setup_python.py setup.py
    pip3 install --no-index -U -v .
}

# pip3 uninstall -y kungfu
pip3 uninstall -y kungfu-python

reinstall 2>err.log >out.log
