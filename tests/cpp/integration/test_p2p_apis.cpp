#include <algorithm>
#include <numeric>
#include <vector>

#include <kungfu.h>

void test_versioned_store(kungfu::Peer &world)
{
    const int rank = world.Rank();
    const int np   = world.Size();

    using T     = int;
    const int n = 10;
    std::vector<T> send_buf(n);
    std::vector<T> recv_buf(n);

    std::iota(send_buf.begin(), send_buf.end(), np);
    std::fill(recv_buf.begin(), recv_buf.end(), -1);

    if (rank == 0) {
        world.Save("v0", "weight", send_buf.data(), send_buf.size(),
                   kungfu::type_encoder::value<T>());
    }

    world.Barrier();

    world.Request(0, "v0", "weight", recv_buf.data(), recv_buf.size(),
                  kungfu::type_encoder::value<T>());

    for (int i = 0; i < n; ++i) {
        if (send_buf[i] != recv_buf[i]) {
            printf("%s failed\n", __func__);
            exit(1);
        }
    }

    world.Barrier();
}

int main(int argc, char *argv[])
{
    kungfu::Peer self;
    test_versioned_store(self);
    return 0;
}
