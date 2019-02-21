FUNCTION(USE_NCCL target)
    TARGET_LINK_LIBRARIES(${target} nccl dl cudart)
ENDFUNCTION()
