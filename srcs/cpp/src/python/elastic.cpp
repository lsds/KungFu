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

    ml::ElasticState es;
    ml::parse_elastic_state(es);

    ds.sync(es.progress());

    std::cout << es.str() << std::endl;

    const int Ki            = 1 << 10;
    int max_sample_per_file = 8 * Ki;
    auto filenames =
        write_tf_record(es, ds, global_batch_size, max_sample_per_file);

    {  // FIXME: return filenames to python
        char name_list_file[256];
        sprintf(name_list_file, "tf-files-from-%d-%d-of-%d.list.txt",
                (int)es.progress(), es.rank(), es.size());
        std::ofstream f(name_list_file);
        for (const auto &filename : filenames) { f << filename << std::endl; }
    }
    return 0;
}
