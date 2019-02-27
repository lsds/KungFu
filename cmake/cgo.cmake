FUNCTION(ADD_CGO_LIBRARY target)
    FILE(MAKE_DIRECTORY ${LIBRARY_OUTPUT_PATH})
    ADD_CUSTOM_TARGET(${target} ALL
                      WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      COMMAND env #
                              CGO_CFLAGS=${CGO_CFLAGS}
                              CGO_LDFLAGS=${CGO_LDFLAGS}
                              go
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
                      WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      COMMAND env #
                              CGO_CFLAGS=${CGO_CFLAGS}
                              CGO_LDFLAGS=${CGO_LDFLAGS}
                              go
                              build
                              -v
                              -buildmode=c-shared
                              -o
                              ${SO_NAME}
                              ${ARGN})
ENDFUNCTION()
