#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "testing.hpp"

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
    kungfu_world _kungfu_world;
    const int n = 100;
    const int m = 100;
    test(n, m);
    return 0;
}
