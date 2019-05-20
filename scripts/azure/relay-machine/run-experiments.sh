#!/bin/sh
set -e

[ -z "${RUNNER}" ] && RUNNER=kungfu
[ -z "${SRC_DIR}" ] && SRC_DIR=$HOME/KungFu
[ -z "${EXPERIMENT_SCRIPT}" ] && EXPERIMENT_SCRIPT=experiments/kungfu/kf_tensorflow_synthetic_benchmark.py
[ -z "${EXPERIMENT_ARGS}" ] && EXPERIMENT_ARGS=

echo "using RUNNER=$RUNNER"
echo "using SRC_DIR=$SRC_DIR"
echo "using EXPERIMENT_SCRIPT=$EXPERIMENT_SCRIPT"
echo "using EXPERIMENT_ARGS=$EXPERIMENT_ARGS"

export PATH=$HOME/local/go/bin:$PATH # TODO: make it default in relay-machine

SCRIPT_NAME=$(basename $0)

show_duration() {
    local ss=$1
    if test $ss -ge 86400; then
        local mm=$((ss / 60))
        local ss=$((ss % 60))

        local hh=$((mm / 60))
        local mm=$((mm % 60))

        local dd=$((hh / 24))
        local hh=$((hh % 24))

        echo "${dd}d${hh}h${mm}m${ss}s"
    elif test $ss -ge 3600; then
        local mm=$((ss / 60))
        local ss=$((ss % 60))

        local hh=$((mm / 60))
        local mm=$((mm % 60))

        echo "${hh}h${mm}m${ss}s"
    elif test $ss -ge 60; then
        local mm=$((ss / 60))
        local ss=$((ss % 60))

        echo "${mm}m${ss}s"
    else
        echo "${ss}s"
    fi
}

measure() {
    local begin=$(date +%s)
    echo "[begin] $SCRIPT_NAME::$@ at $begin" $@
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    local dur=$(show_duration $duration)
    echo "[done] $SCRIPT_NAME::$@ took ${dur}" | tee -a time.log
}

get_host_specs() {
    # FIXME: assuming we are in the internal network, and slots=4 for all machines.
    grep -v '^#' $HOME/hosts.txt | awk '{printf "%s:4:%s\n", $1, $1}'
}

get_host_spec() {
    local H
    for spec in $(get_host_specs); do
        H="${H},${spec}"
    done
    echo $H | cut -b 2-
}

export H=$(get_host_spec)
echo "using H=$H"

gen_ansible_hosts() {
    local H=$1
    for h in $(echo $H | tr ',' '\n'); do
        echo $h | awk -F ':' '{print $3}'
    done
}

# VERBOSE=-v

upload_kungfu() {
    ./scripts/pack.sh
    cp ../KungFu.tar.bz2 .

    gen_ansible_hosts $H >ansible_hosts.txt

    ansible -i ansible_hosts.txt all $VERBOSE -u ${RUNNER} -m file -a 'dest=KungFu state=absent'
    ansible -i ansible_hosts.txt all $VERBOSE -u ${RUNNER} -m unarchive -a 'src=KungFu.tar.bz2 dest=~'
}

init_remote() {
    ansible -i ansible_hosts.txt all $VERBOSE -u ${RUNNER} -m shell -a \
        'pip3 install tensorflow --user'
    ansible -i ansible_hosts.txt all $VERBOSE -u ${RUNNER} -m shell -a \
        ./KungFu/scripts/azure/gpu-machine/install-golang1.11.sh
    ansible -i ansible_hosts.txt all $VERBOSE -u ${RUNNER} -m shell -a \
        'DATA_DIR=$HOME/var/data ./KungFu/scripts/download-mnist.sh'
}

install_remote() {
    ansible -i ansible_hosts.txt all $VERBOSE -u ${RUNNER} -m shell -a \
        'PATH=$HOME/local/go/bin:$PATH pip3 install --user --no-index -U ./KungFu'
}

install_local() {
    ./scripts/go-install.sh --no-tests
}

run_experiments() {
    ./bin/run-experiments -H $H -u ${RUNNER} -timeout 120s \
        env \
        TF_CPP_MIN_LOG_LEVEL=1 \
        python3 \
        ./KungFu/$EXPERIMENT_SCRIPT \
        $EXPERIMENT_ARGS
}

prepare() {
    measure upload_kungfu
    measure install_remote &
    measure install_local &
    wait
}

main() {
    if [ "$1" = "prepare" ]; then
        measure prepare
    elif [ "$1" = "run" ]; then
        measure run_experiments
    elif [ "$1" = "init-remote" ]; then
        measure upload_kungfu
        measure init_remote
    else
        measure prepare
        measure run_experiments
    fi
}

cd $SRC_DIR
measure main $@
