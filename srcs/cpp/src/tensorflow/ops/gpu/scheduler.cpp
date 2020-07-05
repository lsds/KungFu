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

REGISTER_KUNGFU_OP(ResetNcclHelper)
    .Input("in_changed: bool")
    .Input("in_keep: bool")
    .Output("out_changed: bool")
    .Output("out_keep: bool")
    .SetIsStateful();

class ResetNcclHelper : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const bool changed = context->input(0).scalar<bool>()();
        const bool keep    = context->input(1).scalar<bool>()();
        if (changed && keep) { kungfu_python_init_nccl(); }

        Tensor *p_changed = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, MakeTensorShape(),
                                                         &p_changed));
        Tensor *p_keep = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, MakeTensorShape(), &p_keep));
        p_changed->scalar<bool>()() = changed;
        p_keep->scalar<bool>()()    = keep;
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ResetNcclHelper, DEVICE_CPU);
}  // namespace tensorflow
