#include <chrono>
#include <thread>

#include <kungfu/nccl/helper.hpp>
#include <kungfu/tensorflow/ops.h>
#include <kungfu/utils/trace.hpp>
#include <tensorflow/stream_executor/stream.h>

namespace tensorflow
{
void spin_wait(perftools::gputools::Event *event, int ms = 100)
{
    TRACE_SCOPE(__func__);
    while (event->PollForStatus() ==
           perftools::gputools::Event::Status::kPending) {
        std::this_thread::sleep_for(std::chrono::microseconds(ms));
    }
}

perftools::gputools::Event *create_init_ready_event(OpKernelContext *context)
{
    auto device_context = context->op_device_context();
    auto executor       = device_context->stream()->parent();
    auto ready_event    = new perftools::gputools::Event(executor);
    ready_event->Init();
    device_context->stream()->ThenRecordEvent(ready_event);
    return ready_event;
}

void wait_delete_ready_event(perftools::gputools::Event *ready_event)
{
    spin_wait(ready_event);
    delete ready_event;
}

REGISTER_KUNGFU_OP(ScheduledNcclAllReduce)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("op_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class ScheduledNcclAllReduce : public AsyncOpKernel
{
    const KungFu_NCCLScope nccl_scope_;
    std::string op_name_;

  public:
    explicit ScheduledNcclAllReduce(OpKernelConstruction *context)
        : AsyncOpKernel(context), nccl_scope_(KungFu_NCCL_GLOBAL)
    {
        OP_REQUIRES_OK(context, context->GetAttr("op_name", &op_name_));
        OP_REQUIRES(context, op_name_.size() >= 0,
                    errors::InvalidArgument("op_name must not be empty"));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        auto scheduler_  = _default_nccl_helper->EnsureScheduler(nccl_scope_);
        auto controller_ = _default_nccl_helper->EnsureController(nccl_scope_);
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        const auto w     = make_workspace(input, output);
        auto ready_event = create_init_ready_event(context);
        scheduler_->Start(op_name_, [=] {
            wait_delete_ready_event(ready_event);
            controller_->AllReduce(w, KungFu_SUM, done);
        });
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ScheduledNcclAllReduce, DEVICE_GPU);

REGISTER_KUNGFU_OP(NcclAllReduce)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class NcclAllReduce : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        auto scheduler_ =
            _default_nccl_helper->EnsureScheduler(KungFu_NCCL_GLOBAL);
        auto controller_ =
            _default_nccl_helper->EnsureController(KungFu_NCCL_GLOBAL);
        auto peer = _default_peer.get();
        scheduler_->Do([=] { controller_->InitOnce(peer); });
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        wait_delete_ready_event(create_init_ready_event(context));
        controller_->AllReduce(make_workspace(input, output), KungFu_SUM, done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(NcclAllReduce, DEVICE_GPU);

REGISTER_KUNGFU_OP(ScheduledHierarchicalNcclAllReduce)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("op_names: list(string)")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class ScheduledHierarchicalNcclAllReduce : public AsyncOpKernel
{
    const KungFu_NCCLScope nccl_scope_;

    std::string reduce_op_;
    std::string bcast_op_;

  public:
    explicit ScheduledHierarchicalNcclAllReduce(OpKernelConstruction *context)
        : AsyncOpKernel(context), nccl_scope_(KungFu_NCCL_LOCAL)
    {
        std::vector<std::string> op_names;
        OP_REQUIRES_OK(context, context->GetAttr("op_names", &op_names));
        OP_REQUIRES(context, op_names.size() == 2,
                    errors::InvalidArgument("op_names.size() must be 2"));
        reduce_op_ = op_names[0];
        bcast_op_  = op_names[1];
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        auto scheduler_  = _default_nccl_helper->EnsureScheduler(nccl_scope_);
        auto controller_ = _default_nccl_helper->EnsureController(nccl_scope_);
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        auto ready_event  = create_init_ready_event(context);
        auto w_reduce     = make_workspace(input, output);
        auto w_all_reduce = make_workspace(*output, output);
        auto w_bcast      = make_workspace(*output, output);
        auto peer         = _default_peer.get();
        scheduler_->Start(reduce_op_, [=] {
            wait_delete_ready_event(ready_event);
            controller_->Reduce(w_reduce, KungFu_SUM, [=] {
                CrossAllReduceGpu(peer, w_all_reduce, KungFu_SUM, name(), [=] {
                    scheduler_->Start(bcast_op_, [=] {
                        controller_->Broadcast(w_bcast, done);
                    });
                });
            });
        });
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ScheduledHierarchicalNcclAllReduce, DEVICE_GPU);
}  // namespace tensorflow
