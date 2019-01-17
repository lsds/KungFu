OPTION(KUNGFU_BUILD_GTEST "Build gtest from source." OFF)

FUNCTION(LINK_KUNGFU_LIBS target)
    TARGET_LINK_LIBRARIES(${target} kungfu-base kungfu-comm kungfu)
ENDFUNCTION()

FUNCTION(USE_INSTALLED_GTEST target)
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${GTEST_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(${target} ${GTEST_BOTH_LIBRARIES} Threads::Threads)
ENDFUNCTION()

IF(KUNGFU_BUILD_GTEST)
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

    FUNCTION(USE_BUILT_GTEST target)
        TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${PREFIX}/include)
        TARGET_LINK_LIBRARIES(${target} gtest Threads::Threads)
        ADD_DEPENDENCIES(${target} libgtest-dev-repo)
    ENDFUNCTION()

    FUNCTION(ADD_UNIT_TEST target)
        ADD_EXECUTABLE(${target} ${ARGN} ${KUNGFU_TESTS_DIR}/unit/main.cpp)
        LINK_KUNGFU_LIBS(${target})
        USE_BUILT_GTEST(${target})
        ADD_TEST(NAME ${target} COMMAND ${target})
    ENDFUNCTION()
ELSE()
    FIND_PACKAGE(GTest REQUIRED)

    FUNCTION(ADD_UNIT_TEST target)
        ADD_EXECUTABLE(${target} ${ARGN} ${KUNGFU_TESTS_DIR}/unit/main.cpp)
        LINK_KUNGFU_LIBS(${target})
        USE_INSTALLED_GTEST(${target})
        ADD_TEST(NAME ${target} COMMAND ${target})
    ENDFUNCTION()
ENDIF()

SET(KUNGFU_TESTS_DIR ${CMAKE_SOURCE_DIR}/srcs/cpp/tests)

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
    TARGET_LINK_LIBRARIES(${target} Threads::Threads)
    LINK_KUNGFU_LIBS(${target})
    USE_STDTRACER(${target})
ENDFUNCTION()

ADD_TEST_BIN(fake-agent ${KUNGFU_TESTS_DIR}/integration/fake_agent.cpp)
