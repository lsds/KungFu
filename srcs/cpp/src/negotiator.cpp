#include <negotiator.h>

#include <thread>

#include <tensorflow/core/framework/op_kernel.h>

#include <kungfu.h>

class _kungfu_t
{

  public:
    _kungfu_t() { KungfuInit(); }

    ~_kungfu_t() { KungfuFinalize(); }
};

static _kungfu_t _kungfu_world;

namespace tensorflow
{

int to_mpi_type(const DataType &dtype)
{
    switch (dtype) {
    case DT_FLOAT:
        return MPI_FLOAT;
    default:
        // TODO: add more types
        throw std::invalid_argument("unsupported dtype");
    }
}

class Negotiator : public AsyncOpKernel
{
  public:
    explicit Negotiator(OpKernelConstruction *context) : AsyncOpKernel(context)
    {
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));

        // TODO use kungfu::partial_reduce
        KungfuNegotiate(input.tensor_data().data(),
                        (void *)(output->tensor_data().data()),
                        input.NumElements(), to_mpi_type(input.dtype()),
                        MPI_SUM, name().c_str());

        done();  // TODO: call it async
    }
};

REGISTER_KERNEL_BUILDER(Name("Negotiator").Device(DEVICE_CPU), Negotiator);

}  // namespace tensorflow
