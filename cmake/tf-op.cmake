SET(PYTHON "python3" CACHE STRING "python command to use")

IF(NOT DEFINED TF_INCLUDE)
    EXECUTE_PROCESS(
        COMMAND ${PYTHON} -c
                "import tensorflow as tf; print(tf.sysconfig.get_include())"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE TF_INCLUDE)
ENDIF()

IF(NOT DEFINED TF_LIB)
    EXECUTE_PROCESS(
        COMMAND ${PYTHON} -c
                "import tensorflow as tf; print(tf.sysconfig.get_lib())"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE TF_LIB)
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

LINK_DIRECTORIES(${TF_LIB})

FUNCTION(SET_TF_CXX11_ABI target)
    TARGET_COMPILE_OPTIONS(${target} PRIVATE "-D_GLIBCXX_USE_CXX11_ABI=0")
ENDFUNCTION()

FUNCTION(ADD_TF_OP_LIB target)
    ADD_LIBRARY(${target} SHARED ${ARGN})
    TARGET_LINK_LIBRARIES(${target} tensorflow_framework)
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${TF_INCLUDE})
    SET_TF_CXX11_ABI(${target})
    SET_TARGET_PROPERTIES(${target} PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
    SET_TARGET_PROPERTIES(${target} PROPERTIES PREFIX "")
ENDFUNCTION()
