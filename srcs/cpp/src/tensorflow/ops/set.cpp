#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu/mst.hpp>
#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{
REGISTER_OP("HasMember")
    .Attr("T: {int64}")
    .Input("list: T")
    .Input("element: T")
    .Output("has_member: bool");

class HasMember : public OpKernel
{
    using OpKernel::OpKernel;

  public:

    void Compute(OpKernelContext *context) override
    {
        using T = int64_t;
        const auto list = context->input(0).vec<T>();
        const T element = context->input(1).scalar<T>()();
        
        Tensor *output         = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &output));

        output->scalar<bool>()() = false;
        for (int i = 0; i < list.size(); ++i) {
            if(list(i) == element) {
                output->scalar<bool>()() = true;
                break;
            }
        }

        std::cout << "The list of steps is: " << std::endl;
        for (int i = 0; i < list.size(); ++i) {
            std::cout << list(i) << " "; 
        }
        std::cout << std::endl;

        std::cout << "Looking up: " << element << ", Result: " << output->scalar<bool>()() <<  std::endl;

    }
};

// TODO: use macro to add name prefix
REGISTER_KERNEL_BUILDER(Name("HasMember").Device(DEVICE_CPU),
                        HasMember);

}  // namespace tensorflow
