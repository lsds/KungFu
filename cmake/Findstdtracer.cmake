INCLUDE(ExternalProject)

SET(STDTRACER_GIT_URL
    https://github.com/lgarithm/stdtracer.git
    CACHE STRING "URL for clone stdtracer")

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(
    stdtracer-repo
    GIT_REPOSITORY ${STDTRACER_GIT_URL}
    GIT_TAG v0.1.0
    PREFIX ${PREFIX}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PREFIX} -DBUILD_TESTS=0
               -DBUILD_EXAMPLES=0)

ADD_LIBRARY(trace srcs/cpp/src/trace.cpp)
TARGET_INCLUDE_DIRECTORIES(trace PRIVATE ${PREFIX}/src/stdtracer-repo/include)

FUNCTION(USE_STDTRACER target)
    ADD_DEPENDENCIES(${target} stdtracer-repo)
    TARGET_INCLUDE_DIRECTORIES(${target}
                               PRIVATE ${PREFIX}/src/stdtracer-repo/include)
    TARGET_LINK_LIBRARIES(${target} trace)
ENDFUNCTION()
