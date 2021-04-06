CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(kungfu)

SET(CMAKE_CXX_STANDARD 14)

ADD_COMPILE_OPTIONS(-fPIC)

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

OPTION(KUNGFU_BUILD_TESTS "Build tests." OFF)
OPTION(KUNGFU_BUILD_TF_OPS "Build tensorflow operators." OFF)
OPTION(KUNGFU_BUILD_TOOLS "Build kungfu tools." OFF)
OPTION(KUNGFU_ENABLE_AFFINITY "Enable affinity." OFF)
OPTION(KUNGFU_ENABLE_HWLOC "Enable hwloc." OFF)
OPTION(KUNGFU_ENABLE_FLOAT16 "Enable float16." ON)
OPTION(KUNGFU_ENABLE_TRACE "Enable trace." OFF)

INCLUDE(srcs/cmake/strict.cmake)
INCLUDE(srcs/cmake/go.cmake)
IF(KUNGFU_ENABLE_TRACE)
    ADD_DEFINITIONS(-DKUNGFU_ENABLE_TRACE)
    FIND_PACKAGE(stdtracer REQUIRED)
ENDIF()

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/srcs/cpp/include)
LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})

IF(KUNGFU_ENABLE_FLOAT16)
    SET(CGO_CFLAGS "-DENABLE_F16 -mf16c -mavx")
ENDIF()

INCLUDE(srcs/cmake/cgo.cmake)
ADD_CGO_LIBRARY(libkungfu-comm ${CMAKE_SOURCE_DIR}/srcs/go/libkungfu-comm)

ADD_LIBRARY(
    kungfu SHARED srcs/cpp/src/kungfu.cpp srcs/cpp/src/numa/placement.cpp
                  srcs/cpp/src/peer.cpp srcs/cpp/src/session.cpp)
IF(KUNGFU_ENABLE_AFFINITY)
    ADD_DEFINITIONS(-DKUNGFU_ENABLE_AFFINITY)
    TARGET_SOURCES(kungfu PRIVATE srcs/cpp/src/numa/affinity.cpp)
    IF(KUNGFU_ENABLE_HWLOC)
        ADD_DEFINITIONS(-DKUNGFU_ENABLE_HWLOC)
        TARGET_SOURCES(kungfu PRIVATE srcs/cpp/src/numa/hwloc.cpp)
        TARGET_LINK_LIBRARIES(kungfu hwloc)
    ENDIF()
ENDIF()
TARGET_LINK_LIBRARIES(kungfu kungfu-comm)
ADD_DEPENDENCIES(kungfu libkungfu-comm)
ADD_CGO_DEPS(kungfu)

INSTALL(DIRECTORY srcs/cpp/include DESTINATION .)
INSTALL(DIRECTORY lib DESTINATION .)

IF(KUNGFU_ENABLE_NCCL)
    INCLUDE(srcs/cmake/nccl.cmake)
    ADD_LIBRARY(
        kungfu_nccl
        # BEGIN sort
        srcs/cpp/src/cuda/stream.cpp
        srcs/cpp/src/nccl/common.cpp
        srcs/cpp/src/nccl/controller.cpp
        srcs/cpp/src/nccl/gpu_collective.cpp
        srcs/cpp/src/nccl/helper.cpp
        srcs/cpp/src/nccl/scheduler.cpp
        srcs/cpp/src/python/cuda.cpp
        # END sort
    )
    TARGET_LINK_LIBRARIES(kungfu_nccl kungfu)
    USE_NCCL(kungfu_nccl)
    TARGET_COMPILE_DEFINITIONS(kungfu_nccl PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
ENDIF()

IF(KUNGFU_BUILD_TF_OPS)
    INCLUDE(srcs/cmake/tf-op.cmake)

    ADD_LIBRARY(
        kungfu_python SHARED
        srcs/cpp/src/nccl/common.cpp srcs/cpp/src/nccl/scheduler.cpp
        srcs/cpp/src/python/c_api.cpp srcs/cpp/src/python/cuda.cpp
        srcs/cpp/src/python/init.cpp)
    TARGET_LINK_LIBRARIES(kungfu_python kungfu)

    FILE(GLOB CPU_OP_SRCS
         ${CMAKE_SOURCE_DIR}/srcs/cpp/src/tensorflow/ops/cpu/*.cpp)
    FILE(GLOB GPU_OP_SRCS
         ${CMAKE_SOURCE_DIR}/srcs/cpp/src/tensorflow/ops/gpu/*.cpp)

    ADD_TF_OP_LIB(kungfu_tensorflow_ops ${CPU_OP_SRCS})
    TARGET_LINK_LIBRARIES(kungfu_tensorflow_ops kungfu_python kungfu)

    IF(KUNGFU_ENABLE_NCCL)
        TARGET_SOURCES(
            kungfu_python
            PRIVATE srcs/cpp/src/python/init_nccl.cpp
                    srcs/cpp/src/cuda/stream.cpp
                    srcs/cpp/src/nccl/controller.cpp
                    srcs/cpp/src/nccl/helper.cpp
                    srcs/cpp/src/nccl/gpu_collective.cpp)
        USE_NCCL(kungfu_python)
        TARGET_SOURCES(kungfu_tensorflow_ops PRIVATE ${GPU_OP_SRCS})
        USE_NCCL(kungfu_tensorflow_ops)
    ENDIF()
    IF(KUNGFU_ENABLE_TRACE)
        USE_STDTRACER(kungfu_python)
        USE_STDTRACER(kungfu_tensorflow_ops)
    ENDIF()

    SET_TF_COMPILE_OPTION(kungfu_python)
ENDIF()

IF(KUNGFU_BUILD_TOOLS)
    FUNCTION(ADD_GO_BINARY target)
        FILE(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        ADD_CUSTOM_TARGET(
            ${target} ALL
            COMMAND
                env #
                GOBIN=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                CGO_CFLAGS=${CGO_CFLAGS} CGO_LDFLAGS=${CGO_LDFLAGS}
                CGO_CXXFLAGS=${CGO_CXXFLAGS} ${GO} install -v ${ARGN})
    ENDFUNCTION()

    FUNCTION(ADD_KUNGFU_GO_BINARY target)
        ADD_GO_BINARY(${target} ${CMAKE_SOURCE_DIR}/srcs/go/cmd/${target})
    ENDFUNCTION()

    ADD_KUNGFU_GO_BINARY(kungfu-run)
ENDIF()

IF(KUNGFU_BUILD_TESTS)
    ENABLE_TESTING()
    INCLUDE(srcs/cmake/tests.cmake)
ENDIF()
