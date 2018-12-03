#!/bin/sh

set -e

mkdir -p $HOME/local && cd $HOME/local

# https://golang.org/dl/
if [ $(uname -s) = "Darwin" ]; then
    FILENAME=go1.11.2.darwin-amd64.tar.gz
else
    FILENAME=go1.11.2.linux-amd64.tar.gz
fi

[ ! -f $FILENAME ] && curl -vLOJ https://dl.google.com/go/$FILENAME
tar -xf $FILENAME
