#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "testing.hpp"

#include <kungfu/mst.hpp>

DEFINE_TRACE_CONTEXTS;

void test_AllReduce(kungfu::Peer &world, int np)
{
    TRACE_SCOPE(__func__);
    using T     = int32_t;
    const int n = np * 4;
    std::vector<T> x(n);
    std::vector<T> y(n);
    const auto dtype = kungfu::type_encoder::value<T>();

    std::iota(x.begin(), x.end(), 0);

    Waiter waiter;
    world.AllReduce(x.data(), y.data(), n, dtype, KungFu_SUM, "test-tensor",
                    [&waiter] { waiter.done(); });
    waiter.wait();

    int failed = 0;
    for (int i = 0; i < n; ++i) {
        const int expected = i * np;
        if (y[i] != expected) {
            printf("expected y[%d]=%d, but got %d\n", i, expected, y[i]);
            ++failed;
        }
    }
    if (failed) {
        printf("reduce %d elements among %d agents, %d elements failed\n", n,
               np, failed);
        exit(1);
    }
}

void bench_Reduce(kungfu::Peer &world, int n, int m)
{
    TRACE_SCOPE(__func__);

    std::vector<int32_t> x(n);
    std::vector<int32_t> y(n);
    const auto dtype = kungfu::type_encoder::value<int32_t>();
    const std::string name("fake_data");

    for (int i = 0; i < m; ++i) {
        TRACE_SCOPE("KungfuReduce(async)");
        Waiter waiter;
        // void *recvBuf = is_root ? y.data() : nullptr;
        void *recvBuf = y.data();  // TODO: allow nullptr for non-root
        world.Reduce(x.data(), recvBuf, n, dtype, KungFu_SUM, name.c_str(),
                     [&waiter] { waiter.done(); });
        waiter.wait();
    }
}

void bench_AllReduce(kungfu::Peer &world, int n, int m)
{
    TRACE_SCOPE(__func__);

    std::vector<int32_t> x(n);
    std::vector<int32_t> y(n);
    const auto dtype = kungfu::type_encoder::value<int32_t>();
    const std::string name("fake_data");

    for (int i = 0; i < m; ++i) {
        TRACE_SCOPE("KungfuAllReduce(async)");
        Waiter waiter;
        world.AllReduce(x.data(), y.data(), n, dtype, KungFu_SUM, name.c_str(),
                        [&waiter] { waiter.done(); });
        waiter.wait();
    }
}

void test_Gather(kungfu::Peer &world, int m)
{
    const int np       = world.Size();
    const int rank     = world.Rank();
    const bool is_root = rank == 0;

    std::vector<int32_t> x(m);
    std::fill(x.begin(), x.end(), rank);
    if (is_root) {
        std::vector<int32_t> xs(m * np);
        world.Gather(x.data(), x.size(), KungFu_INT32,    //
                     xs.data(), xs.size(), KungFu_INT32,  //
                     "test-gather");
        const int sum = std::accumulate(xs.begin(), xs.end(), 0);
        if (sum != m * np * (np - 1) / 2) {
            printf("invalid Gather result\n");
            exit(1);
        }
    } else {
        world.Gather(x.data(), x.size(), KungFu_INT32,  //
                     nullptr, 0, KungFu_INT32,          //
                     "test-gather");
    }
}

template <typename T1, typename T2> class fake_transform
{
  public:
    void operator()(const T1 *input, int n1, T2 *output, int n2) const
    {
        for (int i = 0; i < n2; ++i) {
            output[i] = (i + 1) * std::accumulate(input, input + n1, 0);
        }
    }
};

void test_AllGatherTransform(kungfu::Peer &world)
{
    const int rank = world.Rank();
    const int np   = world.Size();

    const int m = 10;
    const int n = 3;
    // input
    using T1 = int32_t;
    using T2 = int32_t;
    std::vector<T1> x(m);
    std::fill(x.begin(), x.end(), rank);
    std::vector<T2> y(n);
    std::fill(y.begin(), y.end(), 0);
    world.AllGatherTransform(x.data(), x.size(), y.data(), y.size(),
                             "test-AllGatherTransform",
                             fake_transform<T1, T2>());
    for (int i = 0; i < n; ++i) {
        const int result = (i + 1) * m * np * (np - 1) / 2;
        if (y[i] != result) {
            printf("invalid AllGatherTransform result: %d, expect: %d\n", y[i],
                   result);
            exit(1);
        }
    }
}

void test_MST(kungfu::Peer &world)
{
    const int rank = world.Rank();
    const int np   = world.Size();

    using Weight = float;
    using Vertex = int32_t;
    std::vector<Weight> weights(np);
    for (int i = 0; i < np; ++i) {
        // FIXME: use measured latency as weights
        weights[i] = std::abs(i - rank);
    }
    std::vector<Vertex> edges(2 * (np - 1));
    world.AllGatherTransform(
        weights.data(), weights.size(), edges.data(), edges.size(),  //
        "test-mst", [](const Weight *w, int n, Vertex *v, int m) {
            const int n_vertices = m / 2 + 1;
            if ((n != n_vertices * n_vertices) || (m != 2 * (n_vertices - 1))) {
                throw std::logic_error("invalid input: (" + std::to_string(n) +
                                       "," + std::to_string(m) + ")");
            }
            kungfu::MinimumSpanningTree<Weight, Vertex> mst;
            mst(n_vertices, w, v);
        });

    for (int i = 0; i < np - 1; ++i) {
        printf("(%d, %d)\n", edges[i * 2 + 0], edges[i * 2 + 1]);
    }
}

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    kungfu::Peer _default_peer;
    {
        const int np = _default_peer.Size();
        test_AllReduce(_default_peer, np);
    }
    {
        const int n = 100;
        const int m = 100;
        bench_Reduce(_default_peer, n, m);
    }
    {
        const int n = 100;
        const int m = 100;
        bench_AllReduce(_default_peer, n, m);
    }
    {
        const int m = 100;
        test_Gather(_default_peer, m);
    }
    {
        test_AllGatherTransform(_default_peer);
    }
    {
        test_MST(_default_peer);
    }
    return 0;
}
