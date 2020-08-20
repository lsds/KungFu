OPTION(KUNGFU_BUILD_GTEST "Build gtest from source." OFF)

FIND_PACKAGE(Threads REQUIRED)

FUNCTION(LINK_KUNGFU_LIBS target)
    TARGET_LINK_LIBRARIES(${target} kungfu-comm kungfu)
ENDFUNCTION()

FUNCTION(USE_INSTALLED_GTEST target)
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${GTEST_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(${target} ${GTEST_BOTH_LIBRARIES} Threads::Threads)
ENDFUNCTION()

IF(KUNGFU_BUILD_GTEST)
    INCLUDE(ExternalProject)

    SET(GTEST_GIT_URL
        https://github.com/google/googletest.git
        CACHE STRING "URL for clone gtest")

    SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

    EXTERNALPROJECT_ADD(
        libgtest-dev-repo
        GIT_REPOSITORY ${GTEST_GIT_URL}
        PREFIX ${PREFIX}
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PREFIX}
                   -DCMAKE_CXX_FLAGS=-std=c++11 -Dgtest_disable_pthreads=1
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

SET(KUNGFU_TESTS_DIR ${CMAKE_SOURCE_DIR}/tests/cpp)

FILE(GLOB tests ${KUNGFU_TESTS_DIR}/unit/test_*.cpp)
FOREACH(t ${tests})
    GET_FILENAME_COMPONENT(name ${t} NAME_WE)
    STRING(REPLACE "_" "-" name ${name})
    ADD_UNIT_TEST(${name} ${t})
ENDFOREACH()

FUNCTION(ADD_TEST_BIN target)
    ADD_EXECUTABLE(${target} ${ARGN})
    TARGET_LINK_LIBRARIES(${target} Threads::Threads)
    LINK_KUNGFU_LIBS(${target})
    IF(KUNGFU_ENABLE_TRACE)
        USE_STDTRACER(${target})
    ENDIF()
ENDFUNCTION()

ADD_TEST_BIN(fake-in-proc-trainer
             ${KUNGFU_TESTS_DIR}/integration/fake_in_proc_trainer.cpp)

ADD_TEST_BIN(test-p2p-apis ${KUNGFU_TESTS_DIR}/integration/test_p2p_apis.cpp)
ADD_TEST_BIN(fake-agent ${KUNGFU_TESTS_DIR}/integration/fake_agent.cpp)
ADD_TEST_BIN(fake-kungfu-trainer
             ${KUNGFU_TESTS_DIR}/integration/fake_kungfu_trainer.cpp)

IF(MPI_HOME)
    FIND_PACKAGE(MPI REQUIRED)

    FUNCTION(USE_MPI target)
        TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${MPI_INCLUDE_PATH})
        TARGET_LINK_LIBRARIES(${target} ${MPI_LIBRARIES})
    ENDFUNCTION()

    ADD_TEST_BIN(fake-mpi-trainer
                 ${KUNGFU_TESTS_DIR}/integration/fake_mpi_trainer.cpp)
    USE_MPI(fake-mpi-trainer)
ENDIF()

IF(KUNGFU_ENABLE_NCCL)
    ADD_TEST_BIN(fake-nccl-trainer
                 ${KUNGFU_TESTS_DIR}/integration/fake_nccl_trainer.cpp)
    USE_NCCL(fake-nccl-trainer)
    USE_MPI(fake-nccl-trainer) # FIXME: don't use MPI for bootsrtap
ENDIF()

IF(KUNGFU_BUILD_TF_OPS)
    ADD_TEST_BIN(fake-tf-agent
                 ${KUNGFU_TESTS_DIR}/integration/fake_tf_agent.cpp)
    IF(KUNGFU_ENABLE_NCCL)
        USE_NCCL(fake-tf-agent)
    ENDIF()
    # FIXME: TARGET_LINK_LIBRARIES(fake-tf-agent Threads::Threads)
    TARGET_LINK_LIBRARIES(kungfu Threads::Threads)
    TARGET_LINK_LIBRARIES(fake-tf-agent kungfu_tensorflow_ops)
ENDIF()
