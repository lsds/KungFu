#include <stdml/bits/data/io.hpp>
#include <stdml/bits/data/state2.hpp>
#include <stdml/bits/data/tf_writer.hpp>
#include <stdml/elastic_state.hpp>

#include <fstream>
#include <iostream>
#include <tracer/patient>

namespace stdml::data
{
std::vector<std::string> write_tf_record(const ElasticState &es, state2 ds,
                                         int global_batch_size,
                                         size_t max_sample_per_file)
{

    fprintf(stderr, "%s ...\n", __func__);
    WITH_PATIENT(__func__);

    std::vector<std::string> filenames;

    std::vector<std::string> buffer;

    auto write_buffer = [&]() {
        char filename[256];
        sprintf(filename, "from-%d-worker-%d-of-%d-%d.tf_record",
                (int)es.progress(), es.rank(), es.size(),
                (int)filenames.size());

        std::ofstream f(filename, std::ios::binary);
        for (const auto &text : buffer) {
            f.write(text.data(), text.size());
        }
        filenames.push_back(filename);
        buffer.clear();
    };

    for (;;) {
        YIELD_PATIENT("...");
        auto [batch_idx, total] =
            ds.get_shard(es.rank(), es.size(), global_batch_size);
        if (batch_idx.len() == 0) {
            break;
        }

        auto batch = ds[batch_idx];
        std::cout << "loaded " << batch.size() << " tf records" << std::endl;

        if (buffer.size() > 0 &&
            buffer.size() + batch.size() > max_sample_per_file) {
            write_buffer();
        }

        buffer.insert(buffer.end(), batch.begin(), batch.end());

        if (total < global_batch_size) {  // only possible in last batch
            break;
        }
    }

    if (buffer.size() > 0) {
        write_buffer();
    }

    fprintf(stderr, "%s created %d files for %s.\n", __func__,
            (int)filenames.size(), es.str().c_str());

    return filenames;
}
}  // namespace stdml::data
