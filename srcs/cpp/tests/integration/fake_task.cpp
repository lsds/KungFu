#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "testing.hpp"

class _kungfu_t
{
    static std::string safe_getenv(const char *name)
    {
        const char *ptr = std::getenv(name);
        if (ptr) { return std::string(ptr); }
        return "";
    }

    KungFu_AllReduceAlgo get_algo() const
    {
        const auto value = safe_getenv("KUNGFU_ALLREDUCE_ALGO");
        const std::map<std::string, KungFu_AllReduceAlgo> mp({
            {"SIMPLE", KungFu_SimpleAllReduce},
            {"RING", KungFu_RingAllReduce},
            {"CLIQUE", KungFu_FullSymmetricAllReduce},
            {"TREE", KungFu_TreeAllReduce},
        });
        if (mp.count(value) > 0) { return mp.at(value); }
        return KungFu_SimpleAllReduce;
    }

  public:
    _kungfu_t()
    {
        const auto algo = get_algo();
        KungfuInit(algo);
    }

    ~_kungfu_t() { KungfuFinalize(); }
};

static _kungfu_t _kungfu_world;

void test(int n, int m)
{
    TRACE_SCOPE(__func__);

    std::vector<int32_t> x(n);
    std::vector<int32_t> y(n);
    const auto dtype = kungfu::type_encoder::value<int32_t>();
    std::string name("fake_data");

    for (int i = 0; i < m; ++i) {
        TRACE_SCOPE("KungfuNegotiateAsync");

        std::mutex mu;
        std::condition_variable cv;
        bool done = false;
        KungfuNegotiateAsync(x.data(), y.data(), n, dtype, KungFu_SUM,
                             name.c_str(), [&mu, &cv, &done] {
                                 std::lock_guard<std::mutex> lk(mu);
                                 done = true;
                                 cv.notify_one();
                             });
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&done] { return done; });
        }
    }
    printf("done\n");
}

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    const int n = 100;
    const int m = 100;
    test(n, m);
    return 0;
}
