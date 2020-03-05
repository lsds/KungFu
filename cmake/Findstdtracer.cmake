INCLUDE(ExternalProject)

SET(STDTRACER_GIT_URL
    https://github.com/lgarithm/stdtracer.git
    CACHE STRING "URL for clone stdtracer")

SET(STDTRACER_GIT_TAG
    8c42ecd7bba4c5039db50fedd3b3b190819f859d
    CACHE STRING "Tag for clone stdtracer")

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(
    stdtracer-repo
    GIT_REPOSITORY ${STDTRACER_GIT_URL}
    GIT_TAG ${STDTRACER_GIT_TAG}
    PREFIX ${PREFIX}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PREFIX} -DBUILD_TESTS=0
               -DBUILD_EXAMPLES=0)

ADD_LIBRARY(kungfu_trace srcs/cpp/src/trace.cpp)
TARGET_INCLUDE_DIRECTORIES(kungfu_trace
                           PRIVATE ${PREFIX}/src/stdtracer-repo/include)
ADD_DEPENDENCIES(kungfu_trace stdtracer-repo)

FUNCTION(USE_STDTRACER target)
    TARGET_INCLUDE_DIRECTORIES(${target}
                               PRIVATE ${PREFIX}/src/stdtracer-repo/include)
    TARGET_LINK_LIBRARIES(${target} kungfu_trace)
ENDFUNCTION()
