#!/bin/sh
set -e

cd $(dirname $0)

export KUNGFU_DISABLE_AUTO_LOAD=1

python3 main.py --index 0 &
python3 main.py --index 1 &

wait

echo done
