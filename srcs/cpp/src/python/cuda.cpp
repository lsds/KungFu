#include <algorithm>
#include <sstream>
#include <vector>

#include <kungfu/python/c_api.h>

std::vector<std::string> split(const std::string &s, char sep)
{
    std::vector<std::string> parts;
    std::string part;
    std::istringstream ss(s);
    while (std::getline(ss, part, sep)) {
        if (!part.empty()) { parts.push_back(part); }
    }
    return parts;
}

std::vector<int> parse_cuda_visible_devices(const std::string &val)
{
    const auto parts = split(val, ',');
    std::vector<int> devs;
    for (const auto &p : parts) {
        const int idx = std::stoi(p);
        if (idx >= 0) { devs.push_back(idx); }
    }
    return devs;
}

int kungfu_get_cuda_index()
{
    int dev = 0;
    {
        const char *ptr = std::getenv("KUNGFU_CUDA_VISIBLE_DEVICES");
        if (ptr != nullptr) { dev = std::stoi(ptr); }
    }
    {
        const char *ptr = std::getenv("CUDA_VISIBLE_DEVICES");
        if (ptr != nullptr) {
            const auto devs = parse_cuda_visible_devices(ptr);
            int idx = std::find(devs.begin(), devs.end(), dev) - devs.begin();
            if (idx == static_cast<int>(devs.size())) { idx = -1; }
            return idx;
        } else {
            return dev;
        }
    }
}
