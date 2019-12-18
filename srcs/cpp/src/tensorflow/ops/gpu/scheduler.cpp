#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(StartNcclScheduler).Input("input: string");

class StartNcclScheduler : public OpKernel
{
    int counter_;
    std::vector<int32_t> order_;

    void ResetOrder(int n)
    {
        order_.resize(n);
        std::iota(order_.begin(), order_.end(), 0);
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
        if (names.size() != order_.size()) { ResetOrder(names.size()); }
        if (kungfu::_nccl_order_group.get() != nullptr) {
            if (counter_ == 1) {
                const std::vector<int32_t> arrive_order =
                    kungfu::_nccl_order_group->Wait();
                if (arrive_order.size() == order_.size()) {
                    _kungfu_world->Broadcast(
                        arrive_order.data(), order_.data(), order_.size(),
                        to_kungfu_type(DT_INT32), name().c_str());
                }
            }
        }
        kungfu::_nccl_order_group.reset(new kungfu::order_group(names, order_));
        kungfu::_nccl_controller->InitOnce();
        ++counter_;
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(StartNcclScheduler, DEVICE_CPU);
}  // namespace tensorflow
