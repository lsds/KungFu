# https://cmake.org/cmake/help/v3.9/policy/CMP0068.html
CMAKE_POLICY(SET CMP0068 OLD)

FUNCTION(SET_INSTALL_RPATH target)
    IF(UNIX)
        SET_TARGET_PROPERTIES(${target} PROPERTIES INSTALL_RPATH "$ORIGIN")
        SET_TARGET_PROPERTIES(${target}
                              PROPERTIES BUILD_WITH_INSTALL_RPATH true)
    ENDIF()
    IF(APPLE)
        # FIXME: make it work

        # CMake Warning (dev): Policy CMP0068 is not set: RPATH settings on
        # macOS do not affect install_name.  Run "cmake --help-policy CMP0068"
        # for policy details.  Use the cmake_policy command to set the policy
        # and suppress this warning.
    ENDIF()
ENDFUNCTION()
