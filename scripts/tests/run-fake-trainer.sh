#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)
cd ../..
. ./scripts/utils/measure.sh

KUNGFU_PRUN=$(pwd)/bin/kungfu-prun

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
    env \
        KUNGFU_CONFIG_LOG_CONFIG_VARS=true \
        KUNGFU_TEST_CLUSTER_SIZE=$np \
        ${KUNGFU_PRUN} \
        -np=$np \
        -algo="${ALGO}" \
        -H $H \
        -timeout=120s \
        ./bin/fake-kungfu-trainer
}

run_fake_mpi_trainer() {
    export MPI_HOME=$HOME/local/openmpi
    local np=$1
    $MPI_HOME/bin/mpirun -np $np \
        ./bin/fake-mpi-trainer
}

run_fake_nccl_trainer() {
    local np=$1
    local H=127.0.0.1:$np
    env \
        KUNGFU_CONFIG_LOG_CONFIG_VARS=true \
        KUNGFU_TEST_CLUSTER_SIZE=$np \
        ${KUNGFU_PRUN} \
        -np=$np \
        -H $H \
        -timeout=120s \
        ./bin/fake-nccl-trainer
}

run_in_proc_trainer() {
    local np=$1
    env \
        KUNGFU_TEST_CLUSTER_SIZE=$np \
        ./bin/fake-in-proc-trainer
}

run_fake_trainer_all() {
    local max_np=4
    echo "will $1 with np=1 upto $max_np"
    for np in $(seq $max_np); do
        measure $1 $np
    done
}

main() {
    measure reinstall
    local collective=$1
    if [ -z "$collective" ]; then
        collective=kungfu
    fi

    if [ "$collective" = "kungfu" ]; then
        run_fake_trainer_all run_fake_kungfu_trainer
    elif [ "$collective" = "mpi" ]; then
        run_fake_trainer_all run_fake_mpi_trainer
    elif [ "$collective" = "nccl" ]; then
        run_fake_trainer_all run_fake_nccl_trainer
    elif [ "$collective" = "inproc" ]; then
        run_fake_trainer_all run_in_proc_trainer
    elif [ "$collective" = "all" ]; then
        run_fake_trainer_all run_fake_kungfu_trainer
        run_fake_trainer_all run_fake_mpi_trainer
        # run_fake_trainer_all run_fake_nccl_trainer
        run_fake_trainer_all run_in_proc_trainer
    else
        echo "invalid option"
    fi
}

measure main $@
