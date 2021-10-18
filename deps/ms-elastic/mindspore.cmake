SET(KUNGFU_ROOT ${CMAKE_SOURCE_DIR}/../..)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${KUNGFU_ROOT}/bin)

SET(DEPS_ROOT ${CMAKE_SOURCE_DIR}/deps)
SET(MS_ELASTIC_ROOT ${DEPS_ROOT}/ms-elastic)

FUNCTION(BUILD_MS_ELASTIC_LIB target)
    ADD_LIBRARY(
        ${target}
        ${MS_ELASTIC_ROOT}/src/stdml/data/iter.cpp
        ${MS_ELASTIC_ROOT}/src/stdml/data/state2.cpp
        ${MS_ELASTIC_ROOT}/src/stdml/data/summary.cpp
        ${MS_ELASTIC_ROOT}/src/stdml/data/tf_writer.cpp
        ${MS_ELASTIC_ROOT}/src/stdml/data/tf_index_builder.cpp
        ${MS_ELASTIC_ROOT}/src/stdml/data/index.cpp
        ${MS_ELASTIC_ROOT}/src/stdml/data/io.cpp
        ${MS_ELASTIC_ROOT}/src/stdml/elastic/state.cpp
        ${MS_ELASTIC_ROOT}/src/stdml/utility/stat.cpp
        ${MS_ELASTIC_ROOT}/src/stdml/text/border.cpp)

    TARGET_INCLUDE_DIRECTORIES(
        ${target}
        PRIVATE ${DEPS_ROOT}/ms-elastic/include #
                ${DEPS_ROOT}/stdtracer/include #
                ${DEPS_ROOT}/stdtensor/include #
                ${DEPS_ROOT}/stdnn-ops/include)

    SET_PROPERTY(TARGET ${target} PROPERTY CXX_STANDARD 17)
ENDFUNCTION()

FUNCTION(ADD_MS_ELASTIC_INCLUDE target)
    TARGET_INCLUDE_DIRECTORIES(
        ${target}
        PRIVATE ${CMAKE_SOURCE_DIR}/include
                ${KUNGFU_ROOT}/deps/stdtracer/include
                ${KUNGFU_ROOT}/deps/stdtensor/include
                ${KUNGFU_ROOT}/deps/stdnn-ops/include)
ENDFUNCTION()

FUNCTION(BUILD_MS_ELASTIC_CMDS)
    ADD_EXECUTABLE(ms-elastic-build-tf-index
                   ${CMAKE_SOURCE_DIR}/cmds/ms-elastic-build-tf-index.cpp)

    ADD_MS_ELASTIC_INCLUDE(ms-elastic-build-tf-index)
    SET_PROPERTY(TARGET ms-elastic-build-tf-index PROPERTY CXX_STANDARD 17)
    TARGET_LINK_LIBRARIES(ms-elastic-build-tf-index ms-elastic Threads::Threads)

    # debug tools
    ADD_EXECUTABLE(ms-elastic-create-tf-records
                   ${CMAKE_SOURCE_DIR}/cmds/ms-elastic-create-tf-records.cpp)
    ADD_MS_ELASTIC_INCLUDE(ms-elastic-create-tf-records)
    SET_PROPERTY(TARGET ms-elastic-create-tf-records PROPERTY CXX_STANDARD 17)
    TARGET_LINK_LIBRARIES(ms-elastic-create-tf-records ms-elastic
                          Threads::Threads)

    ADD_EXECUTABLE(ms-read-tf-records
                   ${CMAKE_SOURCE_DIR}/cmds/ms-read-tf-records.cpp)
    ADD_MS_ELASTIC_INCLUDE(ms-read-tf-records)
    SET_PROPERTY(TARGET ms-read-tf-records PROPERTY CXX_STANDARD 17)
    TARGET_LINK_LIBRARIES(ms-read-tf-records ms-elastic Threads::Threads)
ENDFUNCTION()

FIND_PACKAGE(Threads REQUIRED)
BUILD_MS_ELASTIC_LIB(ms-elastic)
# BUILD_MS_ELASTIC_CMDS()
