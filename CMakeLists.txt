CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(kungfu)

SET(CMAKE_CXX_STANDARD 14)

ADD_DEFINITIONS(-Wfatal-errors)
ADD_DEFINITIONS(-Wall)
ADD_COMPILE_OPTIONS(-fPIC)

OPTION(KUNGFU_BUILD_TESTS "Build tests." OFF)
OPTION(KUNGFU_BUILD_TF_OPS "Build tensorflow operators." OFF)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/srcs/cpp/include)
LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})

ADD_LIBRARY(kungfu-base srcs/cpp/src/kungfu_types.cpp
            srcs/cpp/src/kungfu_base.cpp)

INCLUDE(cmake/cgo.cmake)
SET(CGO_CFLAGS "-I${CMAKE_SOURCE_DIR}/srcs/cpp/include")
SET(CGO_LDFLAGS "-L${LIBRARY_OUTPUT_PATH} -lkungfu-base -lstdc++")
ADD_CGO_LIBRARY(libkungfu-comm ${CMAKE_SOURCE_DIR}/srcs/go/libkungfu-comm)
ADD_DEPENDENCIES(libkungfu-comm kungfu-base)

ADD_LIBRARY(kungfu SHARED srcs/cpp/src/kungfu.cpp)
TARGET_LINK_LIBRARIES(kungfu kungfu-comm kungfu-base)
ADD_DEPENDENCIES(kungfu libkungfu-comm)
ADD_CGO_DEPS(kungfu)

IF(KUNGFU_USE_NCCL)
    INCLUDE(cmake/nccl.cmake)
ENDIF()

IF(KUNGFU_BUILD_TF_OPS)
    INCLUDE(cmake/tf-op.cmake)
    INCLUDE(cmake/rpath.cmake)

    ADD_LIBRARY(kungfu_tensorflow_init
                SHARED
                srcs/cpp/src/tensorflow/init.cpp
                srcs/cpp/src/kungfu_comm_go.cpp)
    ADD_LIBRARY(kungfu_tensorflow_init SHARED srcs/cpp/src/tensorflow/init.cpp)
    TARGET_LINK_LIBRARIES(kungfu_tensorflow_init kungfu)

    ADD_TF_OP_LIB(kungfu_tensorflow_ops #
                  srcs/cpp/src/tensorflow/ops/common.cpp
                  srcs/cpp/src/tensorflow/ops/collective.cpp
                  srcs/cpp/src/tensorflow/ops/ako.cpp)
    TARGET_LINK_LIBRARIES(kungfu_tensorflow_ops kungfu_tensorflow_init kungfu)

    IF(KUNGFU_USE_NCCL)
        TARGET_SOURCES(kungfu_tensorflow_init
                       PRIVATE
                       srcs/cpp/src/tensorflow/init_gpu.cpp
                       srcs/cpp/src/kungfu_nccl_comm.cpp)
                       srcs/cpp/src/kungfu_gpu_collective_nccl.cpp)
        USE_NCCL(kungfu_tensorflow_init)
        TARGET_SOURCES(kungfu_tensorflow_ops PRIVATE
                       srcs/cpp/src/tensorflow/ops/collective_gpu.cpp)
    ENDIF()

    SET_TF_CXX11_ABI(kungfu_tensorflow_init)
    SET_INSTALL_RPATH(kungfu_tensorflow_init)
    SET_INSTALL_RPATH(kungfu_tensorflow_ops)
ENDIF()

IF(KUNGFU_BUILD_TESTS)
    ENABLE_TESTING()
    INCLUDE(cmake/tests.cmake)
ENDIF()
