#include <string>
#include <vector>

#include <stdml/bits/data/index.hpp>
#include <stdml/data/tf_writer>
#include <stdml/elastic_state.hpp>

namespace ml = stdml;
namespace md = ml::data;

void usage(int argc, char *argv[])
{
    fprintf(stderr, "Usage:\n\t%s <index-file> <global-batch-size>\n", argv[0]);
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        usage(argc, argv);
        return -1;
    }

    int global_batch_size;
    if (sscanf(argv[2], "%d", &global_batch_size) != 1) {
        usage(argc, argv);
        return -1;
    }

    std::cout << "using global_batch_size: " << global_batch_size << std::endl;

    std::string index_file = argv[1];
    auto index = md::load_total_index(index_file);

    std::cout << index.stat() << std::endl;

    int seed = 0;
    md::state2 ds(index_file, seed);

    ml::ElasticState e;
    ml::parse_elastic_state(e);

    std::cout << e.str() << std::endl;

    const int Ki = 1 << 10;
    int max_sample_per_file = 8 * Ki;
    auto filenames =
        write_tf_record(e, ds, global_batch_size, max_sample_per_file);

    for (const auto &f : filenames) {
        std::cout << f << std::endl;
    }
    return 0;
}
