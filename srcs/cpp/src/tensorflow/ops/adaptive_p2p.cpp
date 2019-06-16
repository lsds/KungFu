#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

#include "peer_selector.hpp"

namespace tensorflow
{

REGISTER_OP("AdaptiveRequestVariables")
    .Attr("T: {float32}")
    .Attr("dtype: type")  // FIXME: infer dtype from T
    .Attr("shapes: list(shape)")
    .Attr("names: list(string)")
    .Attr("ranks: list(int)")
    .Attr("NumTensors: int")
    .Input("vars: NumTensors * T")
    .Output("outputs: NumTensors * T")
    .SetShapeFn(shape_inference::UnchangedShape);

class AdaptiveRequestVariables : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

    int num_tensors_;
    DataType dtype_;
    std::vector<TensorShapeProto> shapes_;
    std::vector<std::string> names_;
    std::vector<int> ranks_;
    std::unique_ptr<AdaptivePeerSelector> peer_selector_;

  public:
    explicit AdaptiveRequestVariables(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("NumTensors", &num_tensors_));
        OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
        OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
        OP_REQUIRES_OK(context, context->GetAttr("names", &names_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));
        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks must not be empty"));
        peer_selector_.reset(new AdaptivePeerSelector(ranks_));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        std::vector<Tensor *> outputs(num_tensors_);
        for (int i = 0; i < num_tensors_; i++) {
            OP_REQUIRES_OK_ASYNC(
                context, context->allocate_output(i, shapes_[i], &outputs[i]),
                done);
        }
        peer_selector_->Do([&](int destination) {
            for (int i = 0; i < num_tensors_; i++) {
                Tensor &t = *outputs.at(i);
                _kungfu_world->Request(
                    destination, names_.at(i).c_str(),
                    const_cast<char *>(t.tensor_data().data()), t.NumElements(),
                    to_kungfu_type(t.dtype()));
            }
        });
        done();
    }
};

REGISTER_KERNEL_BUILDER(Name("AdaptiveRequestVariables").Device(DEVICE_CPU),
                        AdaptiveRequestVariables);

}  // namespace tensorflow
