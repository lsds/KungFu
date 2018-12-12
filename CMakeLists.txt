CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(kungfu)

SET(CMAKE_CXX_STANDARD 14)

ADD_DEFINITIONS(-Wfatal-errors)
ADD_DEFINITIONS(-Wall)
ADD_COMPILE_OPTIONS(-D_GLIBCXX_USE_CXX11_ABI=0)
ADD_COMPILE_OPTIONS(-fPIC)

FIND_PACKAGE(Threads REQUIRED)

INCLUDE(cmake/tf-op.cmake)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/srcs/cpp/include)
LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})

ADD_LIBRARY(algo srcs/cpp/src/algo.cpp)
ADD_LIBRARY(callback srcs/cpp/src/callback.cpp)
ADD_LIBRARY(mpi-types srcs/cpp/src/mpi_types.cpp)

SET(CGO_CFLAGS "-I${CMAKE_SOURCE_DIR}/srcs/cpp/include")
SET(CGO_LDFLAGS
    "-L${LIBRARY_OUTPUT_PATH} -lalgo -lmpi-types -lcallback -lstdc++")

ADD_CUSTOM_TARGET(libgo-kungfu ALL
                  WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                  COMMAND env #
                          CGO_CFLAGS=${CGO_CFLAGS}
                          CGO_LDFLAGS=${CGO_LDFLAGS}
                          go
                          build
                          -v
                          -buildmode=c-archive
                          ${CMAKE_SOURCE_DIR}/srcs/go/libgo-kungfu)
ADD_DEPENDENCIES(libgo-kungfu algo mpi-types callback)

ADD_LIBRARY(kungfu SHARED srcs/cpp/src/kungfu.cpp)
TARGET_INCLUDE_DIRECTORIES(kungfu PRIVATE ${LIBRARY_OUTPUT_PATH})
TARGET_LINK_LIBRARIES(kungfu go-kungfu algo mpi-types callback)
IF(APPLE)
    TARGET_LINK_LIBRARIES(kungfu "-framework CoreFoundation")
    TARGET_LINK_LIBRARIES(kungfu "-framework Security")
ENDIF()
ADD_DEPENDENCIES(kungfu libgo-kungfu)

ADD_TF_OP_LIB(kungfu_tensorflow_ops srcs/cpp/src/negotiator.cpp)
TARGET_LINK_LIBRARIES(kungfu_tensorflow_ops kungfu)
SET_TARGET_PROPERTIES(kungfu_tensorflow_ops PROPERTIES BUILD_RPATH "$ORIGIN")

ADD_CUSTOM_TARGET(kungfu-run ALL
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/srcs/go
                  COMMAND env #
                          GOBIN=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                          go
                          install
                          -v
                          ${CMAKE_SOURCE_DIR}/srcs/go/kungfu-run)

ENABLE_TESTING()
# TODO: add tests
