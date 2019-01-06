INCLUDE(ExternalProject)

SET(GTEST_GIT_URL https://github.com/google/googletest.git
    CACHE STRING "URL for clone gtest")

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(libgtest-dev-repo
                    GIT_REPOSITORY
                    ${GTEST_GIT_URL}
                    PREFIX
                    ${PREFIX}
                    CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${PREFIX}
                    -DCMAKE_CXX_FLAGS=-std=c++11
                    -Dgtest_disable_pthreads=1
                    -DBUILD_GMOCK=0)

LINK_DIRECTORIES(${PREFIX}/lib)

SET(KUNGFU_TESTS_DIR ${CMAKE_SOURCE_DIR}/srcs/cpp/tests)

FUNCTION(ADD_UNIT_TEST target)
    ADD_EXECUTABLE(${target} ${ARGN} ${KUNGFU_TESTS_DIR}/unit/main.cpp)
    TARGET_LINK_LIBRARIES(${target}
                          gtest
                          kungfu-base
                          go-kungfu
                          kungfu
                          Threads::Threads)
    ADD_DEPENDENCIES(${target} libgtest-dev-repo)
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${PREFIX}/include)
    ADD_TEST(NAME ${target} COMMAND ${target})
ENDFUNCTION()

FILE(GLOB tests ${KUNGFU_TESTS_DIR}/unit/test_*.cpp)
FOREACH(t ${tests})
    GET_FILENAME_COMPONENT(name ${t} NAME_WE)
    STRING(REPLACE "_"
                   "-"
                   name
                   ${name})
    ADD_UNIT_TEST(${name} ${t})
ENDFOREACH()

FIND_PACKAGE(stdtracer REQUIRED)

FUNCTION(ADD_TEST_BIN target)
    ADD_EXECUTABLE(${target} ${ARGN} ${KUNGFU_TESTS_DIR}/integration/trace.cpp)
    TARGET_LINK_LIBRARIES(${target}
                          kungfu-base
                          go-kungfu
                          kungfu
                          Threads::Threads)
    ADD_DEPENDENCIES(${target} stdtracer-repo libgo-kungfu)
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${PREFIX}/include)
ENDFUNCTION()

ADD_TEST_BIN(fake-task ${KUNGFU_TESTS_DIR}/integration/fake_task.cpp)
