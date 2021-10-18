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

    namespace md = stdml::data;
    auto index = md::build_total_index(filenames);

    std::cout << index.stat() << std::endl;

    return 0;
}
