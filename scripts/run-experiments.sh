#!/bin/sh
set -e

cd $(dirname $0)/..

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
    echo "begin $@"
    local begin=$(date +%s)
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    local dur=$(show_duration $duration)
    echo "$@ took ${dur}" | tee -a time.log
}

get_host_specs() {
    az vm list-ip-addresses -o table | grep test-cluster-node | awk '{printf "%s:4:%s\n", $3, $2}'
}

get_host_spec() {
    local H=
    for spec in $(get_host_specs); do
        H="${H},${spec}"
    done
    echo $H | cut -b 2-
}

export H=$(get_host_spec)
echo "using H=$H"

gen_hosts() {
    echo "" >hosts.txt
    for h in $(echo $H | tr ',' '\n'); do
        echo $h | awk -F ':' '{print $3}' | cat >>hosts.txt
    done
    cat hosts.txt
}

# VERBOSE=-v
RUNNER=kungfu

upload_kungfu() {
    ./scripts/pack.sh
    cp ../kungfu.tar.bz2 .

    gen_hosts
    ansible -i hosts.txt all $VERBOSE -u ${RUNNER} -m file -a 'dest=kungfu state=absent'
    ansible -i hosts.txt all $VERBOSE -u ${RUNNER} -m unarchive -a 'src=kungfu.tar.bz2 dest=~'
}

install_remote() {
    ansible -i hosts.txt all $VERBOSE -u kungfu -m shell -a \
        'PATH=$HOME/local/go/bin:$PATH pip3 install -v -U ./kungfu'
}

install_local() {
    ./configure && make
    ./scripts/go-install.sh
}

run_experiments() {
    ./bin/run-experiments -H $H -u ${RUNNER} -timeout 120s \
        env \
        TF_CPP_MIN_LOG_LEVEL=1 \
        python3 \
        ./kungfu/benchmarks/tensorflow_synthetic_benchmark.py
}

prepare() {
    measure upload_kungfu
    measure install_remote
    measure install_local
}

main() {
    measure prepare
    measure run_experiments
}

measure main
