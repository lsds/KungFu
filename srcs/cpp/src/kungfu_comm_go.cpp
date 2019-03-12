#include <kungfu_comm.hpp>

namespace kungfu
{

class go_comm : public async_communicator
{
  public:
    go_comm() {}

    ~go_comm() {}

    void all_reduce(const void *send_buf, void *recv_buf, size_t count,
                    KungFu_Datatype dtype, const char *name, DoneCallback done)
    {
        done();
    }
};

async_communicator *new_inter_go_comm(kungfu_world &bootstrap)
{
    return new go_comm;
}

}  // namespace kungfu
