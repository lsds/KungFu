#!/bin/sh

kungfu_run_flags_default() {
    echo -q
    echo -logdir logs
    echo -logfile kungfu-run.log
}

kungfu_run_flags_elastic() {
    kungfu_run_flags_default

    echo -w
    echo -builtin-config-port 9100
    echo -config-server http://127.0.0.1:9100/config
}

kungfu_run_n() {
    local np=$1
    shift
    kungfu-run -np $np $(kungfu_run_flags_default) $@
}

kungfu_run_elastic_n() {
    local np=$1
    shift
    kungfu-run -np $np $(kungfu_run_flags_elastic) $@
}

_show_duration() {
    echo "$1s"
}

trace() {
    echo "BEGIN $@"
    local begin=$(date +%s)
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    echo "END $@, took $(_show_duration $duration)"
    echo
    echo
}

# convenient helper for parallel run
prun() {
    local np=$1
    shift
    trace kungfu_run_n $np $@
}

# convenient helper for elastic run
erun() {
    local np=$1
    shift
    trace kungfu_run_elastic_n $np $@
}

# run as single mode
srun() {
    trace $@
}
