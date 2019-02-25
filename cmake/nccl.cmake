FUNCTION(USE_NCCL target)
    IF(NCCL_HOME)
        TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${NCCL_HOME}/include)
        TARGET_LINK_LIBRARIES(${target} ${NCCL_HOME}/lib/libnccl.so)
    ELSE()
        TARGET_LINK_LIBRARIES(${target} nccl)
    ENDIF()
ENDFUNCTION()
