#pragma once
#include <cstdlib>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "bert.hpp"
#include "resnet50_info.hpp"
#include "vgg_info.hpp"

inline std::vector<int> parameter_sizes(const std::string &model, bool fuse)
{
    static const std::map<std::string, std::vector<int>> model_sizes({
        {"resnet50-imagenet", resnet50_grad_sizes()},
        {"vgg16-imagenet", vgg16_grad_sizes()},
        {"bert", bert_grad_sizes()},
    });
    if (model_sizes.count(model) == 0) {
        fprintf(stderr, "invalid model name: %s\n", model.c_str());
        exit(1);
    }
    std::vector<int> sizes = model_sizes.at(model);
    if (fuse) {
        const int tot = std::accumulate(sizes.begin(), sizes.end(), 0);
        sizes         = {tot};
    }
    return sizes;
}
