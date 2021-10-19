// serve TFrecord shards as a virtual filesystem

#include <iostream>
#include <stdml/bits/vfs/fuse.hpp>

#include <stdml/bits/data/index.hpp>
#include <stdml/bits/data/state2.hpp>
#include <stdml/bits/vfs/tf_record_shard.hpp>
#include <stdml/data/tf_writer>
#include <stdml/elastic_state.hpp>

namespace ml   = stdml;
namespace mlfs = ml::vfs;
namespace md   = ml::data;

struct arguments {
    static void usage(int argc, char *argv[])
    {
        std::vector<std::string> params = {
            "index-file",         //
            "root",               //
            "seed",               //
            "progress",           //
            "cluster-size",       //
            "global-batch-size",  //
            "max-sample-per-file",
        };

        std::stringstream ss;
        ss << "Usage:" << std::endl;
        ss << '\t' << argv[0];
        for (auto p : params) { ss << " <" << p << '>'; }
        ss << std::endl;
        std::cerr << ss.str() << std::endl;
    }

    std::string index_file;
    std::string root;

    int seed;
    int64_t progress;
    int cluster_size;

    int global_batch_size;
    int max_sample_per_file;

    arguments()
        : seed(0), progress(0), cluster_size(1), global_batch_size(1),
          max_sample_per_file(8 * 1024)
    {
    }

    void parse(int argc, char *argv[])
    {
        if (argc != 8) {
            usage(argc, argv);
            exit(1);
        }
        index_file          = argv[1];
        root                = argv[2];
        seed                = std::stoi(argv[3]);
        progress            = std::stoi(argv[4]);
        cluster_size        = std::stoi(argv[5]);
        global_batch_size   = std::stoi(argv[6]);
        max_sample_per_file = std::stoi(argv[7]);
    }
};

int main(int argc, char *argv[])
{
    arguments a;
    a.parse(argc, argv);

    auto index = md::load_total_index(a.index_file);  // keep it in ram
    md::state2 ds(a.index_file, a.seed);
    mlfs::fuse::run(
        a.root,
        [&](mlfs::tree &r) {
            md::mount_tf_index(r, index);
            r.touch("/total.txt", [&] {
                std::stringstream ss;
                ss << index.stat().rows() << std::endl;
                return ss.str();
            }());
            r.touch("/seed.txt", [&] {
                std::stringstream ss;
                ss << a.seed << std::endl;
                return ss.str();
            }());
            // $ cat ./progress-<p>-cluster-of-<n>.txt
            md::mount_tf_record_for_progress(
                r, index, a.progress, a.cluster_size, ds, a.global_batch_size,
                a.max_sample_per_file);  //
        },
        [&] { printf("starting %s on %s\n", argv[0], a.root.c_str()); });
    return 0;
}
