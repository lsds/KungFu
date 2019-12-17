#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(StartNcclScheduler).Input("input: string");

class StartNcclScheduler : public OpKernel
{
    int counter_;
    std::vector<int32_t> permu;

    void ResetOrder(int n)
    {
        permu.resize(n);
        std::iota(permu.begin(), permu.end(), 0);
    }

  public:
    explicit StartNcclScheduler(OpKernelConstruction *context)
        : OpKernel(context), counter_(0)
    {
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        const auto t_names  = input.vec<std::string>();
        std::vector<std::string> names;
        for (int i = 0; i < t_names.size(); ++i) {
            names.push_back(t_names(i));
        }
        if (names.size() != permu.size()) { ResetOrder(names.size()); }
        if (kungfu::_nccl_order_group.get() != nullptr) {
            if (counter_ == 1) {
                const std::vector<int32_t> arrive_order =
                    kungfu::_nccl_order_group->Wait();
                if (arrive_order.size() == permu.size()) {
                    _kungfu_world->Broadcast(
                        arrive_order.data(), permu.data(), permu.size(),
                        to_kungfu_type(DT_INT32), name().c_str());
                }
            }
        }
        kungfu::_nccl_order_group.reset(new kungfu::order_group(names, permu));
        ++counter_;
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(StartNcclScheduler, DEVICE_CPU);
}  // namespace tensorflow
