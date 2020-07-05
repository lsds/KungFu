#include <kungfu/nccl/helper.hpp>
#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(StartNcclScheduler)
    .Attr("scope: string")
    .Input("input: string");

class StartNcclScheduler : public OpKernel
{
    KungFu_NCCLScope nccl_scope_;

  public:
    explicit StartNcclScheduler(OpKernelConstruction *context)
        : OpKernel(context)
    {
        std::string scope_name;
        OP_REQUIRES_OK(context, context->GetAttr("scope", &scope_name));
        OP_REQUIRES(context, kungfu::_nccl_scopes.count(scope_name) > 0,
                    errors::InvalidArgument("invalid scope"));
        nccl_scope_ = kungfu::_nccl_scopes.at(scope_name);
    }

    void Compute(OpKernelContext *context) override
    {
        auto scheduler_ = _default_nccl_helper->EnsureScheduler(nccl_scope_);
        const Tensor &input = context->input(0);
        const auto t_names  = input.vec<std::string>();
        std::vector<std::string> names;
        for (int i = 0; i < t_names.size(); ++i) {
            names.push_back(t_names(i));
        }
        scheduler_->Reset(names);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(StartNcclScheduler, DEVICE_CPU);

REGISTER_KUNGFU_OP(ResetNcclHelper).SetIsStateful();

class ResetNcclHelper : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        kungfu_python_init_nccl();
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ResetNcclHelper, DEVICE_CPU);
}  // namespace tensorflow
