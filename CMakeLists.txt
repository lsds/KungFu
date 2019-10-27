CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(kungfu)

SET(CMAKE_CXX_STANDARD 14)

ADD_DEFINITIONS(-Wfatal-errors)
ADD_DEFINITIONS(-Wall)
ADD_COMPILE_OPTIONS(-fPIC)

OPTION(KUNGFU_BUILD_TESTS "Build tests." OFF)
OPTION(KUNGFU_BUILD_TF_OPS "Build tensorflow operators." OFF)
OPTION(KUNGFU_BUILD_TOOLS "Build kungfu tools." OFF)
OPTION(KUNGFU_ENABLE_FLOAT16 "Enable float16." ON)

IF($ENV{READTHEDOCS})
    # https://docs.readthedocs.io/en/stable/builds.html#the-build-environment
    OPTION(KUNGFU_DOWNLOAD_GO "Download golang." ON)
ELSE()
    OPTION(KUNGFU_DOWNLOAD_GO "Download golang." OFF)
ENDIF()

IF(KUNGFU_DOWNLOAD_GO)
    FILE(DOWNLOAD "https://dl.google.com/go/go1.13.3.linux-amd64.tar.gz"
         ${CMAKE_SOURCE_DIR}/go1.13.3.linux-amd64.tar.gz
         SHOW_PROGRESS EXPECTED_MD5 e0b36adf4dbb7fa53b477df5d7b1dd8c)
    EXECUTE_PROCESS(COMMAND tar -xf
                            ${CMAKE_SOURCE_DIR}/go1.13.3.linux-amd64.tar.gz)
    SET(GO "${CMAKE_SOURCE_DIR}/go/bin/go")
ELSE()
    FIND_PROGRAM(GO NAMES go)
ENDIF()

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/srcs/cpp/include)
LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})

IF(KUNGFU_ENABLE_FLOAT16)
    SET(CGO_CFLAGS "-DENABLE_F16 -mf16c -mavx")
ENDIF()

INCLUDE(cmake/cgo.cmake)
ADD_CGO_LIBRARY(libkungfu-comm ${CMAKE_SOURCE_DIR}/srcs/go/libkungfu-comm)

ADD_LIBRARY(kungfu SHARED srcs/cpp/src/kungfu.cpp)
TARGET_LINK_LIBRARIES(kungfu kungfu-comm)
ADD_DEPENDENCIES(kungfu libkungfu-comm)
ADD_CGO_DEPS(kungfu)

IF(KUNGFU_ENABLE_NCCL)
    INCLUDE(cmake/nccl.cmake)
ENDIF()

IF(KUNGFU_BUILD_TF_OPS)
    INCLUDE(cmake/tf-op.cmake)

    ADD_LIBRARY(kungfu_python SHARED srcs/cpp/src/python/init.cpp)
    TARGET_LINK_LIBRARIES(kungfu_python kungfu)

    ADD_TF_OP_LIB(kungfu_tensorflow_ops
                  # BEGIN sort
                  srcs/cpp/src/tensorflow/ops/collective.cpp
                  srcs/cpp/src/tensorflow/ops/control.cpp
                  srcs/cpp/src/tensorflow/ops/local.cpp
                  srcs/cpp/src/tensorflow/ops/p2p_new.cpp
                  srcs/cpp/src/tensorflow/ops/peer_to_peer.cpp
                  srcs/cpp/src/tensorflow/ops/topology.cpp
                  # END sort
                  )
    TARGET_LINK_LIBRARIES(kungfu_tensorflow_ops kungfu_python kungfu)

    IF(KUNGFU_ENABLE_NCCL)
        TARGET_SOURCES(kungfu_python
                       PRIVATE
                       srcs/cpp/src/python/init_gpu.cpp
                       srcs/cpp/src/kungfu_gpu_collective_nccl.cpp)
        USE_NCCL(kungfu_python)
        TARGET_SOURCES(kungfu_tensorflow_ops PRIVATE
                       srcs/cpp/src/tensorflow/ops/collective_gpu.cpp)
    ENDIF()

    SET_TF_CXX11_ABI(kungfu_python)
ENDIF()

IF(KUNGFU_BUILD_TOOLS)
    FUNCTION(ADD_GO_BINARY target)
        FILE(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        ADD_CUSTOM_TARGET(${target} ALL
                          COMMAND env #
                                  GOBIN=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                                  CGO_CFLAGS=${CGO_CFLAGS}
                                  CGO_LDFLAGS=${CGO_LDFLAGS}
                                  CGO_CXXFLAGS=${CGO_CXXFLAGS}
                                  ${GO}
                                  install
                                  -v
                                  ${ARGN})
    ENDFUNCTION()

    FUNCTION(ADD_KUNGFU_GO_BINARY target)
        ADD_GO_BINARY(${target} ${CMAKE_SOURCE_DIR}/srcs/go/cmd/${target})
    ENDFUNCTION()

    ADD_KUNGFU_GO_BINARY(kungfu-run)
ENDIF()

IF(KUNGFU_BUILD_TESTS)
    ENABLE_TESTING()
    INCLUDE(cmake/tests.cmake)
ENDIF()
