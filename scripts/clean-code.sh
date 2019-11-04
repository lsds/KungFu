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
    jq -r '.[].file' compile_commands.json
}

list_hdr_and_srcs() {
    find ./srcs -type f | grep .cpp$
    find ./srcs -type f | grep .hpp$
    find ./srcs -type f | grep .h$
}

list_py_srcs() {
    find ./srcs -type f | grep .py$
    # find ./examples -type f | grep .py$
    # find ./tests -type f | grep .py$
    # find ./benchmarks -type f | grep .py$
}

for_all() {
    for src in $(list_srcs); do
        echo "$1 $src"
        $1 $src
    done
}

fmt_all_cpp() {
    for src in $(list_hdr_and_srcs); do
        echo "clang-format -i $src"
        clang-format -i $src
    done
}

check_all() {
    for_all check
}

fix_all() {
    for_all fix
}

fmt_py() {
    # autoflake -i --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys $1
    autoflake -i $1
    isort -y $1
    yapf -i $1
}

fmt_all_py() {
    for src in $(list_py_srcs); do
        echo "fmt_py $src"
        fmt_py $src
    done
}

main() {
    case $1 in
    --fmt-cpp)
        fmt_all_cpp
        ;;
    --fmt-py)
        fmt_all_py
        ;;
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
