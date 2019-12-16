#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(StartNcclScheduler).Input("input: string");

std::string show_list(const std::vector<int32_t> &a)
{
    std::string line;
    for (auto x : a) { line += ", " + std::to_string(x); }
    return line;
}

class StartNcclScheduler : public OpKernel
{
    int counter_;
    std::vector<int32_t> permu;

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
        if (counter_ > 0) {
            if (counter_ == 1) {
                const std::vector<int32_t> arrive_order =
                    kungfu::_nccl_order_group->Wait();
                // const auto line = show_list(arrive_order);
                // fprintf(stderr, "arrive_order|%d| %s\n", counter_,
                // line.c_str());
                if (arrive_order.size() == permu.size()) {
                    _kungfu_world->Broadcast(
                        arrive_order.data(), permu.data(), permu.size(),
                        to_kungfu_type(DT_INT32), name().c_str());
                }
            }
        } else {
            permu.resize(names.size());
            std::iota(permu.begin(), permu.end(), 0);
        }
        kungfu::_nccl_order_group.reset(new kungfu::order_group(names, permu));
        ++counter_;
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(StartNcclScheduler, DEVICE_CPU);
}  // namespace tensorflow
