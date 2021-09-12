#pragma once
#include <optional>

#include <stdml/bits/collective/peer.hpp>

namespace stdml::collective
{
class state3
{
    peer peer_;
    std::unique_ptr<session> sess_;
    std::optional<int64_t> max_progress_;

    int64_t progress_;
    bool synced_;
    bool detached_;

    peer::config_prodiver get_config_;

    using init_state_fn = std::function<void(const session &, int64_t)>;
    using sync_state_fn = std::function<void(session &, int64_t)>;

    init_state_fn init_state_;
    sync_state_fn sync_state_;

    bool sync_progress();
    void check_resize();

    void sync();

    void set_max_progress(int64_t n);

    friend int64_t
    elastic_run3(const std::function<int64_t(const session &, int64_t)> &init,
                 const std::function<void(session &, int64_t)> &sync,
                 const std::function<int64_t(session &, int64_t)> &run);

  public:
    state3(peer::config_prodiver get_config, init_state_fn init_state,
           sync_state_fn sync_state, std::optional<int64_t> max_progress = {});

    // TODO: implement builtin config_prodiver in C++
    // elastic_state(std::optional<int64_t> max_progress = {});

    operator int64_t();

    void operator+=(int64_t progress);
    void operator++();

    bool should_stop();

    session &sess();
};
}  // namespace stdml::collective
