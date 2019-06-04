#!/bin/sh
set -e

add_bin_path_after() {
    export PATH=$PATH:$1/bin
}

add_bin_path_after $HOME/local/clang

cd $(dirname $0)/..

rebuild() {
    ./configure --verbose --build-tensorflow-ops
    make
}

check() {
    clang-tidy "$1"
}

fix() {
    clang-tidy -fix "$1"
}

list_srcs() {
    # find ./srcs -type f | grep .cpp$
    jq -r '.[].file' compile_commands.json
}

for_all() {
    for src in $(list_srcs); do
        echo "$1 $src"
        $1 $src
    done
}

check_all() {
    for_all check
}

fix_all() {
    for_all fix
}

main() {
    case $1 in
    --check)
        rebuild
        check_all
        ;;
    --fix)
        rebuild
        fix_all
        ;;
    '')
        rebuild
        check_all
        ;;
    *)
        echo "unknown argument $1"
        exit 1
        ;;
    esac
}

main $@
