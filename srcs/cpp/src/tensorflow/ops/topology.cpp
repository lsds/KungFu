#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu/mst.hpp>
#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{
REGISTER_OP("KungfuMinimumSpanningTree")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("weight: T")
    .Output("edges: int32");

class MinimumSpanningTree : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &weights = context->input(0);
        const int n           = weights.NumElements();
        Tensor *edges         = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context,
            context->allocate_output(0, MakeTensorShape(n - 1, 2), &edges),
            done);
        using Weight = float;  // FIXME: use type switch on T
        _kungfu_world->AllGatherTransform(
            weights.vec<Weight>().data(), n,                                  //
            const_cast<int32 *>(edges->matrix<int32>().data()), 2 * (n - 1),  //
            "mst",
            [n = n](const Weight *w, int /* n*n */, int32_t *v,
                    int /* 2(n-1) */) {
                kungfu::MinimumSpanningTree<Weight, int32> mst;
                mst(n, w, v);
            });
        done();
    }
};

// TODO: use macro to add name prefix
REGISTER_KERNEL_BUILDER(Name("KungfuMinimumSpanningTree").Device(DEVICE_CPU),
                        MinimumSpanningTree);

}  // namespace tensorflow
