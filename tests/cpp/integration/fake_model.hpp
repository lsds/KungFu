#pragma once
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "resnet50_info.hpp"
#include "vgg_info.hpp"

inline std::vector<int> parameter_sizes(const std::string &model, bool fuse)
{
    std::vector<int> sizes;
    if (model == std::string("resnet50-imagenet")) {
        sizes = resnet50_grad_sizes();
    } else if (model == std::string("vgg16-imagenet")) {
        sizes = vgg16_grad_sizes();
    } else {
        fprintf(stderr, "invalid model name: %s\n", model.c_str());
        exit(1);
    }
    if (fuse) {
        const int tot = std::accumulate(sizes.begin(), sizes.end(), 0);
        sizes         = {tot};
    }
    return sizes;
}
