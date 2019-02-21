#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)
cd ../..
. ./scripts/utils/measure.sh

reinstall() {
    CMAKE_SOURCE_DIR=$(pwd)
    export CGO_CFLAGS="-I${CMAKE_SOURCE_DIR}/srcs/cpp/include"
    export CGO_LDFLAGS="-L${CMAKE_SOURCE_DIR}/lib -lkungfu-base -lstdc++"

    ./scripts/go-install.sh
}

run_fake_kungfu_trainer() {
    # local ALGO=
    local np=$1
    local H=127.0.0.1:$np
    KUNGFU_TEST_CLUSTER_SIZE=$np \
        ./bin/kungfu-prun \
        -np=$np \
        -algo="${ALGO}" \
        -H $H \
        -timeout=120s \
        ${QUIET} \
        ./bin/fake-kungfu-trainer
}

run_fake_mpi_trainer() {
    export MPI_HOME=$HOME/local/openmpi
    local np=$1
    $MPI_HOME/bin/mpirun -np $np \
        ./bin/fake-mpi-trainer
}

run_fake_trainer_all() {
    local max_np=4
    echo "will test $1 for from np=1 upto np=$max_np"
    for np in $(seq $max_np); do
        measure $1 $np
    done
}

main() {
    measure reinstall
    local max_np=4
    if [ "$1" = "mpi" ]; then
        run_fake_trainer_all run_fake_mpi_trainer
    else
        export KUNGFU_CONFIG_LOG_CONFIG_VARS=true
        run_fake_trainer_all run_fake_kungfu_trainer
    fi
}

main $@
