#include <string>
#include <vector>

#include <stdml/bits/data/index.hpp>

int main(int argc, char *argv[])
{
    std::vector<std::string> filenames;
    for (int i = 1; i < argc; ++i) {
        filenames.emplace_back(argv[i]);
    }
    if (filenames.size() <= 0) {
        fprintf(stderr, "no TFRecord file given\n");
        return 0;
    }

    bool include_meta_bytes = true;

    namespace md = stdml::data;
    auto index = md::build_total_index(filenames);
    std::string idx_file("tf-index-" + std::to_string(filenames.size()) +
                         ".idx.txt");

    std::ofstream f(idx_file);

    if (include_meta_bytes) {
        index.save_with_meta(f);
    } else {
        index.save(f);
    }

    std::cout << "built " << idx_file << std::endl;
    return 0;
}
