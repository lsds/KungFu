#include <thread>

#include <kungfu/tensorflow/ops.h>
#include <tensorflow/stream_executor/stream.h>

namespace tensorflow
{
inline void spin_wait(perftools::gputools::Event *event, int ms = 100)
{
    // TRACE_SCOPE(__func__);
    while (event->PollForStatus() ==
           perftools::gputools::Event::Status::kPending) {
        std::this_thread::sleep_for(std::chrono::microseconds(ms));
    }
}

inline perftools::gputools::Event *
create_init_ready_event(OpKernelContext *context)
{
    auto device_context = context->op_device_context();
    auto executor       = device_context->stream()->parent();
    auto ready_event    = new perftools::gputools::Event(executor);
    ready_event->Init();
    device_context->stream()->ThenRecordEvent(ready_event);
    return ready_event;
}

inline void wait_delete_ready_event(perftools::gputools::Event *ready_event)
{
    spin_wait(ready_event);
    delete ready_event;
}
}  // namespace tensorflow
