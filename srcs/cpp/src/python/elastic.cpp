#include <kungfu.h>
#include <kungfu/python/c_api.h>

#include <string>
#include <vector>

#include <stdml/bits/data/index.hpp>
#include <stdml/data/tf_writer>
#include <stdml/elastic_state.hpp>

namespace ml = stdml;
namespace md = ml::data;

int kungfu_create_tf_records(const char *index_file, int seed,
                             int global_batch_size)
{
    std::cout << "using global_batch_size: " << global_batch_size << std::endl;

    auto index = md::load_total_index(index_file);

    std::cout << index.stat() << std::endl;

    md::state2 ds(index_file, seed);

    ml::ElasticState e;
    ml::parse_elastic_state(e);

    std::cout << e.str() << std::endl;

    const int Ki            = 1 << 10;
    int max_sample_per_file = 8 * Ki;
    auto filenames =
        write_tf_record(e, ds, global_batch_size, max_sample_per_file);

    for (const auto &f : filenames) { std::cout << f << std::endl; }

    return 0;
}
