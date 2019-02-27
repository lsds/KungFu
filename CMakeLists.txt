CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(kungfu)

SET(CMAKE_CXX_STANDARD 14)

ADD_DEFINITIONS(-Wfatal-errors)
ADD_DEFINITIONS(-Wall)
ADD_COMPILE_OPTIONS(-fPIC)

OPTION(KUNGFU_BUILD_TESTS "Build tests." OFF)
OPTION(KUNGFU_BUILD_TF_OPS "Build tensorflow operators." OFF)

FIND_PACKAGE(Threads REQUIRED)

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

TARGET_INCLUDE_DIRECTORIES(kungfu PRIVATE ${LIBRARY_OUTPUT_PATH})
TARGET_LINK_LIBRARIES(kungfu kungfu-comm kungfu-base)
IF(APPLE)
    TARGET_LINK_LIBRARIES(kungfu "-framework CoreFoundation")
    TARGET_LINK_LIBRARIES(kungfu "-framework Security")
ELSE()
    TARGET_LINK_LIBRARIES(kungfu rt)
ENDIF()
ADD_DEPENDENCIES(kungfu libkungfu-comm)

IF(KUNGFU_USE_NCCL)
    INCLUDE(cmake/nccl.cmake)
ENDIF()

FUNCTION(SET_INSTALL_RPATH target)
    SET_TARGET_PROPERTIES(${target} PROPERTIES INSTALL_RPATH "$ORIGIN")
    SET_TARGET_PROPERTIES(${target} PROPERTIES BUILD_WITH_INSTALL_RPATH true)
ENDFUNCTION()

IF(KUNGFU_BUILD_TF_OPS)
    INCLUDE(cmake/tf-op.cmake)

    ADD_LIBRARY(kungfu_python SHARED srcs/cpp/src/kungfu_python.cpp)
    TARGET_LINK_LIBRARIES(kungfu_python kungfu)

    ADD_TF_OP_LIB(kungfu_tensorflow_ops #
                  srcs/cpp/src/tensorflow/ops/common.cpp
                  srcs/cpp/src/tensorflow/ops/all_reduce.cpp)
    TARGET_LINK_LIBRARIES(kungfu_tensorflow_ops kungfu_python kungfu)

    IF(UNIX)
        SET_INSTALL_RPATH(kungfu_python)
        SET_INSTALL_RPATH(kungfu_tensorflow_ops)
    ENDIF()
    IF(APPLE)
        # FIXME: ?
    ENDIF()
ENDIF()

IF(KUNGFU_BUILD_TESTS)
    ENABLE_TESTING()
    INCLUDE(cmake/tests.cmake)
ENDIF()
