#!/bin/sh
set -e

cd $(dirname $0)

./gen-compose.py
docker-compose up
