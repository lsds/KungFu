CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(kungfu)

SET(CMAKE_CXX_STANDARD 14)
ADD_COMPILE_OPTIONS(-D_GLIBCXX_USE_CXX11_ABI=0)
ADD_COMPILE_OPTIONS(-fPIC)

FIND_PACKAGE(Threads REQUIRED)

FUNCTION(TARGET_STATIC_LINK_LIBRARIES target)
    IF(APPLE)
        # NOT required
        TARGET_LINK_LIBRARIES(${target} ${ARGN})
    ELSE()
        TARGET_LINK_LIBRARIES(${target}
                              -Wl,--whole-archive
                              ${ARGN}
                              -Wl,--no-whole-archive)
    ENDIF()
ENDFUNCTION()

INCLUDE(cmake/tf-op.cmake)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

ADD_LIBRARY(algo src/cpp/algo.cpp)
ADD_LIBRARY(mpi-types src/cpp/mpi_types.cpp)

SET(CGO_CFLAGS "-I${CMAKE_SOURCE_DIR}/include")
SET(CGO_LDFLAGS "-L${CMAKE_SOURCE_DIR}/lib -lalgo -lmpi-types -lstdc++")

ADD_CUSTOM_TARGET(go-output-dir ALL
                  COMMAND ${CMAKE_COMMAND} -E make_directory go-output)
ADD_CUSTOM_TARGET(go-kungfu ALL
                  WORKING_DIRECTORY "go-output"
                  COMMAND env #
                          CGO_CFLAGS=${CGO_CFLAGS}
                          CGO_LDFLAGS=${CGO_LDFLAGS}
                          go
                          build
                          -v
                          -buildmode=c-archive
                          ${CMAKE_SOURCE_DIR}/src/go/go-kungfu)
ADD_DEPENDENCIES(go-kungfu go-output-dir algo mpi-types)

ADD_LIBRARY(kungfu SHARED src/cpp/kungfu.cpp)
TARGET_INCLUDE_DIRECTORIES(kungfu PRIVATE ${CMAKE_SOURCE_DIR}/go-output)
TARGET_LINK_LIBRARIES(kungfu
                      ${CMAKE_SOURCE_DIR}/go-output/go-kungfu.a
                      algo
                      mpi-types)
ADD_DEPENDENCIES(kungfu go-kungfu)

ADD_TF_OP_LIB(kungfu_tensorflow_ops src/cpp/negotiator.cpp)
TARGET_LINK_LIBRARIES(kungfu_tensorflow_ops kungfu)

ADD_CUSTOM_TARGET(kungfu-run ALL
                  WORKING_DIRECTORY "."
                  COMMAND env #
                          GOBIN=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                          go
                          install
                          -v
                          ${CMAKE_SOURCE_DIR}/src/go/kungfu-run)

ENABLE_TESTING()
# TODO: add tests
