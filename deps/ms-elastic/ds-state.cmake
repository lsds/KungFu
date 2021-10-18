FUNCTION(add_ds_state_srcs target)
    TARGET_SOURCES(
        ${target}
        PRIVATE ${MS_ELASTIC_ROOT}/src/stdml/data/iter.cpp
                ${MS_ELASTIC_ROOT}/src/stdml/data/state2.cpp
                ${MS_ELASTIC_ROOT}/src/stdml/data/summary.cpp
                ${MS_ELASTIC_ROOT}/src/stdml/data/tf_writer.cpp
                ${MS_ELASTIC_ROOT}/src/stdml/data/tf_index_builder.cpp
                ${MS_ELASTIC_ROOT}/src/stdml/data/index.cpp
                ${MS_ELASTIC_ROOT}/src/stdml/data/io.cpp
                ${MS_ELASTIC_ROOT}/src/stdml/elastic/state.cpp
                ${MS_ELASTIC_ROOT}/src/stdml/utility/stat.cpp
                ${MS_ELASTIC_ROOT}/src/stdml/text/border.cpp)
ENDFUNCTION()
