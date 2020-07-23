#include <kungfu/nccl/common.hpp>
#include <kungfu/ncclv2/helper.hpp>
#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(BeginNcclOps).Attr("scope: string").Input("input: string");

class BeginNcclOps : public OpKernel
{
    KungFu_NCCLScope nccl_scope_;

  public:
    explicit BeginNcclOps(OpKernelConstruction *context) : OpKernel(context)
    {
        std::string scope_name;
        OP_REQUIRES_OK(context, context->GetAttr("scope", &scope_name));
        OP_REQUIRES(context, kungfu::_nccl_scopes.count(scope_name) > 0,
                    errors::InvalidArgument("invalid scope"));
        nccl_scope_ = kungfu::_nccl_scopes.at(scope_name);
    }

    void Compute(OpKernelContext *context) override
    {

        const Tensor &input = context->input(0);
        const auto t_names  = input.vec<std::string>();
        std::vector<std::string> names;
        for (int i = 0; i < t_names.size(); ++i) {
            names.push_back(t_names(i));
        }
        if (nccl_scope_ == KungFu_NCCL_LOCAL) {
            _default_nccl_helper_v2->BeginScheduleHierarchicalAllReduce(names);
        } else {
            _default_nccl_helper_v2->BeginScheduleAllReduce(names);
        }
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(BeginNcclOps, DEVICE_CPU);
}  // namespace tensorflow
