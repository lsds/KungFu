#include <cstdlib>
#include <string>
#include <vector>

#include <kungfu.h>
#include <kungfu/python/c_api.h>

#include <stdml/bits/data/index.hpp>
#include <stdml/data/tf_writer>
#include <stdml/elastic_state.hpp>

namespace ml = stdml;
namespace md = ml::data;

const int Ki = 1 << 10;

int kungfu_create_tf_records(const char *index_file, int seed,
                             int global_batch_size)
{
    int max_sample_per_file = 8 * Ki;
    if (const char *p = std::getenv("MAX_SAMPLE_PER_FILE"); p != nullptr) {
        max_sample_per_file = std::stoi(p);
    }

    std::cout << "using global_batch_size: " << global_batch_size << std::endl;

    auto index = md::load_total_index(index_file);

    std::cout << "global index: " << index.stat() << std::endl;

    md::state2 ds(index_file, seed);

    ml::ElasticState es;
    ml::parse_elastic_state(es);

    ds.sync(es.progress());

    std::cout << es.str() << std::endl;

    auto shard =
        write_tf_record(es, ds, global_batch_size, max_sample_per_file);

    {  // FIXME: return struct to python
        char name_list_file[256];
        sprintf(name_list_file, "tf-files-from-%d-%d-of-%d.list.txt",
                (int)es.progress(), es.rank(), es.size());
        save_shard_result(name_list_file, shard);
    }
    return 0;
}
