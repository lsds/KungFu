SET(PYTHON "python3" CACHE STRING "python command to use")

IF(NOT DEFINED TF_COMPILE_FLAGS)
    EXECUTE_PROCESS(
        COMMAND
            ${PYTHON} -c
            "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE TF_COMPILE_FLAGS)
ENDIF()

IF(NOT DEFINED TF_LINK_FLAGS)
    EXECUTE_PROCESS(
        COMMAND
            ${PYTHON} -c
            "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE TF_LINK_FLAGS)
ENDIF()

IF(NOT DEFINED PY_EXT_SUFFIX)
    EXECUTE_PROCESS(
        COMMAND ${PYTHON} -c
                # sysconfig.get_config_var('EXT_SUFFIX')  does't work for
                # python2
                "import sysconfig; print(sysconfig.get_config_var('SO'))"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE PY_EXT_SUFFIX)
ENDIF()

FUNCTION(SET_TF_COMPILE_OPTION target)
    SET_TARGET_PROPERTIES(${target}
                          PROPERTIES COMPILE_FLAGS ${TF_COMPILE_FLAGS})
ENDFUNCTION()

FUNCTION(ADD_TF_OP_LIB target)
    ADD_LIBRARY(${target} SHARED ${ARGN})
    SET_TF_COMPILE_OPTION(${target})
    TARGET_LINK_LIBRARIES(${target} ${TF_LINK_FLAGS})
    SET_TARGET_PROPERTIES(${target} PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
    SET_TARGET_PROPERTIES(${target} PROPERTIES PREFIX "")
ENDFUNCTION()
