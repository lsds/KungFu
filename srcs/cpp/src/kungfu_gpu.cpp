#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include <kungfu.h>

int kungfu_world::NegotiateGPUAsync(const void *sendbuf, void *recvbuf,
                                    int count, KungFu_Datatype dtype,
                                    KungFu_Op op, const char *name,
                                    DoneCallback done)
{
    const int buffer_size = kungfu_type_size(dtype) * count;
    // TODO: use memory pool
    auto input_cpu  = new std::vector<char>(buffer_size);
    auto output_cpu = new std::vector<char>(buffer_size);
    if (cudaMemcpy(input_cpu->data(), sendbuf, buffer_size,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed");
    }
    return NegotiateAsync(
        input_cpu->data(), output_cpu->data(), count, dtype, op, name,
        [done, input_cpu, output_cpu, recvbuf, buffer_size] {
            if (cudaMemcpy(recvbuf, output_cpu->data(), buffer_size,
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                throw std::runtime_error("cudaMemcpy failed");
            }
            delete input_cpu;
            delete output_cpu;
            done();
        });
}
