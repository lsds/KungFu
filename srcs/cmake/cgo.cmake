SET(CGO_OUTPUT_DIRECTORY
    ${LIBRARY_OUTPUT_PATH}
    CACHE STRING "directory for CGO output")

FUNCTION(ADD_CGO_LIBRARY target)
    FILE(MAKE_DIRECTORY ${LIBRARY_OUTPUT_PATH})
    ADD_CUSTOM_TARGET(${target} ALL
                      WORKING_DIRECTORY ${CGO_OUTPUT_DIRECTORY}
                      COMMAND env #
                              CGO_CFLAGS=${CGO_CFLAGS}
                              CGO_LDFLAGS=${CGO_LDFLAGS}
                              CGO_CXXFLAGS=${CGO_CXXFLAGS}
                              ${GO}
                              build
                              -v
                              -buildmode=c-archive
                              ${ARGN})
ENDFUNCTION()

FUNCTION(ADD_CGO_SHARED_LIBRARY target)
    FILE(MAKE_DIRECTORY ${LIBRARY_OUTPUT_PATH})
    GET_FILENAME_COMPONENT(NAME ${ARGN} NAME)
    SET(SO_NAME "${NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    ADD_CUSTOM_TARGET(${target} ALL
                      WORKING_DIRECTORY ${CGO_OUTPUT_DIRECTORY}
                      COMMAND env #
                              CGO_CFLAGS=${CGO_CFLAGS}
                              CGO_LDFLAGS=${CGO_LDFLAGS}
                              CGO_CXXFLAGS=${CGO_CXXFLAGS}
                              ${GO}
                              build
                              -v
                              -buildmode=c-shared
                              -o
                              ${SO_NAME}
                              ${ARGN})
ENDFUNCTION()

FUNCTION(ADD_CGO_DEPS target)
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${CGO_OUTPUT_DIRECTORY})
    # Some CGO libraries require extra dependencies
    IF(APPLE)
        TARGET_LINK_LIBRARIES(${target} "-framework CoreFoundation")
        TARGET_LINK_LIBRARIES(${target} "-framework Security")
    ELSE()
        TARGET_LINK_LIBRARIES(${target} rt)
    ENDIF()
ENDFUNCTION()
