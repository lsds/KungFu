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
    INCLUDE(cmake/rpath.cmake)

    ADD_LIBRARY(kungfu_tensorflow_init SHARED srcs/cpp/src/tensorflow/init.cpp)
    TARGET_LINK_LIBRARIES(kungfu_tensorflow_init kungfu)

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
    TARGET_LINK_LIBRARIES(kungfu_tensorflow_ops kungfu_tensorflow_init kungfu)

    IF(KUNGFU_ENABLE_NCCL)
        TARGET_SOURCES(kungfu_tensorflow_init
                       PRIVATE
                       srcs/cpp/src/tensorflow/init_gpu.cpp
                       srcs/cpp/src/kungfu_gpu_collective_nccl.cpp)
        USE_NCCL(kungfu_tensorflow_init)
        TARGET_SOURCES(kungfu_tensorflow_ops PRIVATE
                       srcs/cpp/src/tensorflow/ops/collective_gpu.cpp)
    ENDIF()

    SET_TF_CXX11_ABI(kungfu_tensorflow_init)
    SET_INSTALL_RPATH(kungfu_tensorflow_init)
    SET_INSTALL_RPATH(kungfu_tensorflow_ops)
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
                                  go
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
