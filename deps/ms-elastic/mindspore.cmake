# FIXME: don't add global include
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/ms-elastic/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/stdtracer/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/stdtensor/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/stdnn-ops/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/stdml-collective/include)

ADD_LIBRARY(
    ms-elastic
    ${CMAKE_SOURCE_DIR}/ms-elastic/src/stdml/data/iter.cpp
    ${CMAKE_SOURCE_DIR}/ms-elastic/src/stdml/data/state2.cpp
    ${CMAKE_SOURCE_DIR}/ms-elastic/src/stdml/data/summary.cpp
    # ${CMAKE_SOURCE_DIR}/ms-elastic/src/stdml/data/tf_reader.cpp
    ${CMAKE_SOURCE_DIR}/ms-elastic/src/stdml/data/tf_writer.cpp
    ${CMAKE_SOURCE_DIR}/ms-elastic/src/stdml/data/tf_index_builder.cpp
    ${CMAKE_SOURCE_DIR}/ms-elastic/src/stdml/data/index.cpp
    ${CMAKE_SOURCE_DIR}/ms-elastic/src/stdml/data/io.cpp
    ${CMAKE_SOURCE_DIR}/ms-elastic/src/stdml/elastic/state.cpp
    ${CMAKE_SOURCE_DIR}/ms-elastic/src/stdml/utility/stat.cpp
    ${CMAKE_SOURCE_DIR}/ms-elastic/src/stdml/text/border.cpp)

TARGET_INCLUDE_DIRECTORIES(
    ms-elastic
    PRIVATE ${CMAKE_SOURCE_DIR}/ms-elastic/include
            ${CMAKE_SOURCE_DIR}/third_party/stdtracer/include
            ${CMAKE_SOURCE_DIR}/third_party/stdtensor/include
            ${CMAKE_SOURCE_DIR}/third_party/stdnn-ops/include)

SET_PROPERTY(TARGET ms-elastic PROPERTY CXX_STANDARD 17)

ADD_EXECUTABLE(
    ms-elastic-build-tf-index
    ${CMAKE_SOURCE_DIR}/ms-elastic/cmds/ms-elastic-build-tf-index.cpp)

TARGET_INCLUDE_DIRECTORIES(
    ms-elastic-build-tf-index
    PRIVATE ${CMAKE_SOURCE_DIR}/ms-elastic/include
            ${CMAKE_SOURCE_DIR}/third_party/stdtracer/include
            ${CMAKE_SOURCE_DIR}/third_party/stdtensor/include
            ${CMAKE_SOURCE_DIR}/third_party/stdnn-ops/include)

SET_PROPERTY(TARGET ms-elastic-build-tf-index PROPERTY CXX_STANDARD 17)
TARGET_LINK_LIBRARIES(ms-elastic-build-tf-index ms-elastic Threads::Threads)

# debug tools
ADD_EXECUTABLE(
    ms-elastic-create-tf-records
    ${CMAKE_SOURCE_DIR}/ms-elastic/cmds/ms-elastic-create-tf-records.cpp)

TARGET_INCLUDE_DIRECTORIES(
    ms-elastic-create-tf-records
    PRIVATE ${CMAKE_SOURCE_DIR}/ms-elastic/include
            ${CMAKE_SOURCE_DIR}/third_party/stdtracer/include
            ${CMAKE_SOURCE_DIR}/third_party/stdtensor/include
            ${CMAKE_SOURCE_DIR}/third_party/stdnn-ops/include)

SET_PROPERTY(TARGET ms-elastic-create-tf-records PROPERTY CXX_STANDARD 17)
TARGET_LINK_LIBRARIES(ms-elastic-create-tf-records ms-elastic Threads::Threads)

ADD_EXECUTABLE(ms-read-tf-records
               ${CMAKE_SOURCE_DIR}/ms-elastic/cmds/ms-read-tf-records.cpp)

TARGET_INCLUDE_DIRECTORIES(
    ms-read-tf-records
    PRIVATE ${CMAKE_SOURCE_DIR}/ms-elastic/include
            ${CMAKE_SOURCE_DIR}/third_party/stdtracer/include
            ${CMAKE_SOURCE_DIR}/third_party/stdtensor/include
            ${CMAKE_SOURCE_DIR}/third_party/stdnn-ops/include)

SET_PROPERTY(TARGET ms-read-tf-records PROPERTY CXX_STANDARD 17)
TARGET_LINK_LIBRARIES(ms-read-tf-records ms-elastic Threads::Threads)
