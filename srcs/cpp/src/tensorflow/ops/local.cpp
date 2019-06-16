#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{
REGISTER_OP("SaveVariables")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("names: list(string)")
    .Attr("NumTensors: int")
    .Input("vars: NumTensors * T");

class SaveVariables : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

    int num_tensors_;
    std::vector<std::string> names_;

  public:
    explicit SaveVariables(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("NumTensors", &num_tensors_));
        OP_REQUIRES_OK(context, context->GetAttr("names", &names_));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        for (int i = 0; i < num_tensors_; ++i) {
            const Tensor &t = context->input(i);
            // TODO: get name from t
            _kungfu_world->Save(names_[i].c_str(), t.tensor_data().data(),
                                t.NumElements(), to_kungfu_type(t.dtype()));
        }
        done();
    }
};

REGISTER_KERNEL_BUILDER(Name("SaveVariables").Device(DEVICE_CPU),
                        SaveVariables);

}  // namespace tensorflow
