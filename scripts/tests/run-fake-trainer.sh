#!/bin/sh
set -e

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
SCRIPT_NAME=$(dirname $0)
cd ../..
. ./scripts/utils/measure.sh

export MPI_HOME=$HOME/local/openmpi
KUNGFU_RUN=$(pwd)/bin/kungfu-run

reinstall() {
    ./scripts/go-install.sh
    env \
        GOBIN=$(pwd)/bin \
        go install -v ./tests/go/...
}

run_fake_kungfu_trainer() {
    local np=$1
    local H=127.0.0.1:$np
    env \
        KUNGFU_CONFIG_LOG_CONFIG_VARS=true \
        KUNGFU_TEST_CLUSTER_SIZE=$np \
        ${KUNGFU_RUN} \
        -np=$np \
        -H $H \
        -timeout=120s \
        ./bin/fake-kungfu-trainer
}

run_fake_mpi_trainer() {
    local np=$1
    $MPI_HOME/bin/mpirun -np $np \
        ./bin/fake-mpi-trainer
}

run_fake_nccl_trainer() {
    local np=$1
    # $MPI_HOME/bin/mpirun -np $np \
    #     ./bin/fake-nccl-trainer
    local H=127.0.0.1:$np
    env \
        KUNGFU_CONFIG_LOG_CONFIG_VARS=true \
        KUNGFU_TEST_CLUSTER_SIZE=$np \
        ${KUNGFU_RUN} \
        -np=$np \
        -H $H \
        -timeout=120s \
        ./bin/fake-nccl-trainer
}

run_fake_go_trainer() {
    local np=$1
    local H=127.0.0.1:$np
    env \
        KUNGFU_TEST_CLUSTER_SIZE=$np \
        ${KUNGFU_RUN} \
        -np=$np \
        -H $H \
        -timeout=120s \
        ./bin/kungfu-fake-go-trainer

}

installed=

install_pip() {
    if [ -z $installed ]; then
        pip3 install --user -U .
        installed=1
    fi
}

run_fake_tf_trainer() {
    install_pip
    local np=$1
    local H=127.0.0.1:$np

    env \
        KUNGFU_TEST_CLUSTER_SIZE=$np \
        ${KUNGFU_RUN} \
        -np=$np \
        -H $H \
        -timeout=120s \
        python3 \
        ./tests/python/integration/fake_tf_trainer.py
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
    elif [ "$collective" = "go" ]; then
        run_fake_trainer_all run_fake_go_trainer
    elif [ "$collective" = "tf" ]; then
        run_fake_trainer_all run_fake_tf_trainer
    elif [ "$collective" = "inproc" ]; then
        run_fake_trainer_all run_in_proc_trainer
    elif [ "$collective" = "all" ]; then
        run_fake_trainer_all run_fake_kungfu_trainer
        run_fake_trainer_all run_fake_mpi_trainer
        run_fake_trainer_all run_fake_go_trainer
        run_fake_trainer_all run_fake_tf_trainer
        run_fake_trainer_all run_in_proc_trainer
        if [ -f /usr/include/nccl.h ]; then
            run_fake_trainer_all run_fake_nccl_trainer
        fi
    else
        echo "invalid option"
    fi
}

measure main $@
