#!/bin/sh
set -e

cd $(dirname $0)

if [ $(uname -s) = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")
fi

python3 ./pingpong.py
