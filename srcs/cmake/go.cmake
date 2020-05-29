OPTION(KUNGFU_DOWNLOAD_GO "Download golang." OFF)

IF($ENV{READTHEDOCS})
    # https://docs.readthedocs.io/en/stable/builds.html#the-build-environment
    SET(KUNGFU_DOWNLOAD_GO ON)
ENDIF()

IF($ENV{KUNGFU_DOWNLOAD_GO})
    SET(KUNGFU_DOWNLOAD_GO ON)
ENDIF()

IF(KUNGFU_DOWNLOAD_GO)
    FILE(DOWNLOAD "https://dl.google.com/go/go1.13.3.linux-amd64.tar.gz"
         ${CMAKE_SOURCE_DIR}/go1.13.3.linux-amd64.tar.gz
         SHOW_PROGRESS EXPECTED_MD5 e0b36adf4dbb7fa53b477df5d7b1dd8c)
    EXECUTE_PROCESS(COMMAND tar
                            -xf
                            ${CMAKE_SOURCE_DIR}/go1.13.3.linux-amd64.tar.gz
                            -C
                            ${CMAKE_SOURCE_DIR})
    SET(GO "${CMAKE_SOURCE_DIR}/go/bin/go")
ELSE()
    FIND_PROGRAM(GO NAMES go)
ENDIF()
