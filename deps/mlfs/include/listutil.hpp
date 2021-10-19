#include <vector>

template <typename T>
std::vector<std::pair<T, int>> group(const std::vector<T> &xs)
{
    std::vector<std::pair<T, int>> g;
    if (xs.size() <= 0) { return g; }
    T x   = xs[0];
    int c = 1;
    for (size_t i = 1; i < xs.size(); ++i) {
        if (xs[i] != x) {
            g.emplace_back(x, c);
            x = xs[i];
            c = 1;
        } else {
            ++c;
        }
    }
    g.emplace_back(x, c);
    return g;
}
