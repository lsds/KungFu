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

SET(CGO_CFLAGS "-I${CMAKE_SOURCE_DIR}/srcs/cpp/include")
SET(CGO_LDFLAGS "-L${LIBRARY_OUTPUT_PATH} -lkungfu-base -lstdc++")

ADD_CUSTOM_TARGET(libkungfu-comm ALL
                  WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                  COMMAND env #
                          CGO_CFLAGS=${CGO_CFLAGS}
                          CGO_LDFLAGS=${CGO_LDFLAGS}
                          go
                          build
                          -v
                          -buildmode=c-archive
                          ${CMAKE_SOURCE_DIR}/srcs/go/libkungfu-comm)
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

IF(KUNGFU_BUILD_TF_OPS)
    INCLUDE(cmake/tf-op.cmake)
    ADD_TF_OP_LIB(kungfu_tensorflow_ops srcs/cpp/src/kungfu_tensorflow_ops.cpp)
    TARGET_LINK_LIBRARIES(kungfu_tensorflow_ops kungfu)
    SET_TARGET_PROPERTIES(kungfu_tensorflow_ops
                          PROPERTIES INSTALL_RPATH "$ORIGIN")
    SET_TARGET_PROPERTIES(kungfu_tensorflow_ops
                          PROPERTIES BUILD_WITH_INSTALL_RPATH true)
ENDIF()

IF(KUNGFU_BUILD_TESTS)
    ENABLE_TESTING()
    INCLUDE(cmake/tests.cmake)
ENDIF()
