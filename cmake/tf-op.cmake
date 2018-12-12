SET(PYTHON "python3" CACHE STRING "python command to use")

EXECUTE_PROCESS(
    COMMAND ${PYTHON} -c
            "import tensorflow as tf; print(tf.sysconfig.get_include())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TF_INCLUDE)

EXECUTE_PROCESS(COMMAND ${PYTHON} -c
                        "import tensorflow as tf; print(tf.sysconfig.get_lib())"
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE TF_LIB)

EXECUTE_PROCESS(
    COMMAND ${PYTHON} -c
            "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PY_EXT_SUFFIX)

LINK_DIRECTORIES(${TF_LIB})

FUNCTION(ADD_TF_OP_LIB target)
    ADD_LIBRARY(${target} SHARED ${ARGN})
    TARGET_LINK_LIBRARIES(${target} tensorflow_framework)
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${TF_INCLUDE})
    SET_TARGET_PROPERTIES(${target} PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
    SET_TARGET_PROPERTIES(${target} PROPERTIES PREFIX "")
ENDFUNCTION()
