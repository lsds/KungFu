#include <kungfu/tensorflow/ops.h>

std::vector<std::string> split(const std::string &s, char sep)
{
    std::vector<std::string> parts;
    std::string part;
    std::istringstream ss(s);
    while (std::getline(ss, part, sep)) {
        if (!part.empty()) { parts.push_back(part); }
    }
    return parts;
}

namespace tensorflow
{
REGISTER_KUNGFU_OP(StepBasedSchedule)
    .Attr("config: string")
    .Attr("default: int")
    .Attr("strict: bool")
    .Input("step: int32")
    .Output("cluster_size: int32");

class StepBasedSchedule : public OpKernel
{
    using range_t = std::pair<int, int>;

    std::vector<std::pair<range_t, int>> schedule_;
    int32_t default_;
    bool strict_;

  public:
    StepBasedSchedule(OpKernelConstruction *context)
        : OpKernel(context), default_(0), strict_(false)
    {
        std::string config;
        OP_REQUIRES_OK(context, context->GetAttr("config", &config));
        OP_REQUIRES(context, config.size() > 0,
                    errors::InvalidArgument("config can't be empty"));
        const auto parts = split(config, ',');
        int offset       = 0;
        for (const auto &part : parts) {
            const auto kv = split(part, ':');
            OP_REQUIRES(context, kv.size() == 2,
                        errors::InvalidArgument("invalid config"));
            const int k = std::stoi(kv[0]);
            const int v = std::stoi(kv[1]);
            schedule_.push_back(std::make_pair(range_t(offset, offset + v), k));
            offset += v;
        }
        OP_REQUIRES_OK(context, context->GetAttr("default", &default_));
        OP_REQUIRES_OK(context, context->GetAttr("strict", &strict_));
    }

    void Compute(OpKernelContext *context) override
    {
        const int32_t step   = context->input(0).scalar<int32_t>()();
        Tensor *cluster_size = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, MakeTensorShape(),
                                                         &cluster_size));
        bool found = false;
        int result = default_;
        for (const auto &sch : schedule_) {  // FIXME: use binary search
            const auto r = sch.first;
            if (r.first <= step && step < r.second) {
                result = sch.second;
                found  = true;
                break;
            }
        }
        if (strict_) {
            OP_REQUIRES(context, found,
                        errors::InvalidArgument("schedule not found"));
        } else {
            LOG(INFO) << "schedule not found for " << step << ", using default "
                      << result;  // FIXME: infrequently
        }
        cluster_size->scalar<int32_t>()() = result;
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(StepBasedSchedule, DEVICE_CPU);
}  // namespace tensorflow
