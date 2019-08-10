#!/usr/bin/env bash

rm -rf ./p100-logs
mkdir p100-logs
obsutil cp -f -r -include=*stdout.log obs://obs-mnist-ic/logs/$1 ./p100-logs


# do it in python
# iterate through all files, compute sum throughput for all peers at iteration X => output the sum
# show throughput distribution for all peers

