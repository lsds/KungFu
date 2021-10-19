SET(KUNGFU_ROOT ${CMAKE_SOURCE_DIR}/../..)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${KUNGFU_ROOT}/bin)
SET(DEPS_ROOT ${KUNGFU_ROOT}/deps)
SET(MS_ELASTIC_ROOT ${DEPS_ROOT}/ms-elastic)

MESSAGE("KUNGFU_ROOT: ${KUNGFU_ROOT}")
MESSAGE("DEPS_ROOT: ${DEPS_ROOT}")
INCLUDE(${MS_ELASTIC_ROOT}/ds-state.cmake)

FUNCTION(TARGET_SOURCE_TREE target)
    FILE(GLOB_RECURSE SRCS ${ARGN})
    TARGET_SOURCES(${target} PRIVATE ${SRCS})
ENDFUNCTION()

FUNCTION(MLFS_DEPS target)
    SET_PROPERTY(TARGET ${target} PROPERTY CXX_STANDARD 17)
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)

    TARGET_INCLUDE_DIRECTORIES(
        ${target}
        PRIVATE ${DEPS_ROOT}/ms-elastic/include #
                ${DEPS_ROOT}/stdtracer/include #
                ${DEPS_ROOT}/stdtensor/include #
                ${DEPS_ROOT}/stdnn-ops/include #
                ${DEPS_ROOT}/stdml-collective/include #
    )
ENDFUNCTION()

FUNCTION(BUILD_MLFS_LIB target)
    ADD_LIBRARY(${target} SHARED)
    MLFS_DEPS(${target})
    ADD_DS_STATE_SRCS(${target})
    TARGET_SOURCE_TREE(${target} ${CMAKE_SOURCE_DIR}/src/*.cpp)
    TARGET_LINK_LIBRARIES(${target} PRIVATE fuse)
ENDFUNCTION()

FUNCTION(BUILD_TFRECORD_FS target)
    ADD_EXECUTABLE(${target} ${CMAKE_SOURCE_DIR}/cmds/tfrecord-fs.cpp)
    MLFS_DEPS(${target})
    TARGET_LINK_LIBRARIES(${target} PRIVATE ${ARGN})
ENDFUNCTION()
