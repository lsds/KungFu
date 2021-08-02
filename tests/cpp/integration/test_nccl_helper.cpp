#include <iostream>
#include <numeric>

#include "cuda_vector.hpp"
#include <kungfu.h>
#include <kungfu/nccl/helper.hpp>
#include <kungfu/peer.hpp>

int main()
{
    auto &peer = kungfu::Peer::GetDefault();
    std::cout << peer.Rank() << '/' << peer.Size() << std::endl;

    auto &nccl_helper = kungfu::NCCLHelper::GetDefault(true);

    // Test AllReduce
    {
        using R = float;
        int n   = 1024;
        cuda_vector<R> x(n);
        cuda_vector<R> y(n);
        kungfu::CudaStream s;

        kungfu::Workspace w{
            .sendbuf = x.data(),
            .recvbuf = y.data(),
            .count   = n,
            .dtype   = kungfu::type_encoder::value<R>(),
        };

        auto controller = nccl_helper->EnsureController(KungFu_NCCL_GLOBAL);
        controller->InitOnce(&peer);
        controller->AllReduce(w, KungFu_SUM, static_cast<cudaStream_t>(s));
        s.sync();
    }

    // Test SubsetAllReduce
    auto test_SubsetAllReduce = [&](auto topology) {
        using R = float;
        int n   = 1024;

        std::vector<R> x_cpu(n);
        std::vector<R> y_cpu(n);
        cuda_vector<R> x(n);
        cuda_vector<R> y(n);

        kungfu::CudaStream s;
        std::iota(x_cpu.begin(), x_cpu.end(), 1);
        std::fill(y_cpu.begin(), y_cpu.end(), -1);
        s.memcpy(x.data(), x_cpu.data(), n * sizeof(R), cudaMemcpyHostToDevice);
        s.memcpy(y.data(), y_cpu.data(), n * sizeof(R), cudaMemcpyHostToDevice);

        kungfu::Workspace w{
            .sendbuf = x.data(),
            .recvbuf = y.data(),
            .count   = n,
            .dtype   = kungfu::type_encoder::value<R>(),
        };

        auto controller = nccl_helper->EnsureGroupController(topology);
        controller->InitOnce(&peer);
        controller->AllReduce(w, KungFu_SUM, static_cast<cudaStream_t>(s));
        s.sync();
        s.memcpy(y_cpu.data(), y.data(), n * sizeof(R), cudaMemcpyDeviceToHost);
        return y_cpu;
    };

    if (peer.Size() == 4) {
        std::vector<int32_t> topology(4);
        {
            topology[0] = 0;
            topology[1] = 0;
            topology[2] = 0;
            topology[3] = 0;
            auto y      = test_SubsetAllReduce(topology);
            printf("rank=%d, y[0]=%f, groups={[0,1,2,3]}\n", peer.Rank(), y[0]);
        }
        {
            topology[0] = 0;
            topology[1] = 1;
            topology[2] = 1;
            topology[3] = 1;
            auto y      = test_SubsetAllReduce(topology);
            printf("rank=%d, y[0]=%f, groups={[0], [1,2,3]}\n", peer.Rank(),
                   y[0]);
        }
        {
            topology[0] = 0;
            topology[1] = 0;
            topology[2] = 2;
            topology[3] = 2;
            auto y      = test_SubsetAllReduce(topology);
            printf("rank=%d, y[0]=%f, groups={[0,1], [2,3]}\n", peer.Rank(),
                   y[0]);
        }
    }

    nccl_helper.reset(nullptr);
    return 0;
}
