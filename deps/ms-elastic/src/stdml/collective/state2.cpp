#include <stdml/bits/collective/state2.hpp>

/*
J   ->   S  <-> [C]
^        | \
|        v  \
P   <-   R   .-> F
^        |       |
|        v       v
I        D       Q
*/

namespace stdml::collective
{
state2::state2(peer::config_prodiver get_config, init_state_fn init_state,
               sync_state_fn sync_state, std::optional<int64_t> max_progress)
    : peer_(peer::from_env()),
      sess_(peer_.join_elastic()),
      max_progress_(max_progress),
      progress_(0),
      synced_(false),
      detached_(false),
      get_config_(std::move(get_config)),
      //   init_state_(std::move(init_state)),
      sync_state_(std::move(sync_state))
{
    sync_progress();
    if (init_state) {
        init_state(*sess_, progress_);
    }
}

bool state2::sync_progress()
{
    if (!synced_) {
        progress_ = sess_->all_reduce(progress_, max);
        synced_ = true;
        return true;
    }
    return false;
}

void state2::check_resize()
{
    auto result = peer_.resize(sess_, get_config_);
    if (result.detached) {
        detached_ = true;
    }
    if (result.changed) {
        synced_ = false;
    }
}

void state2::sync()
{
    if (sync_progress()) {
        sync_state_(*sess_, progress_);
    }
}

state2::operator int64_t()
{
    if (detached_) {
        return progress_;
    }
    sync();
    return progress_;
}

void state2::operator+=(int64_t progress)
{
    if (detached_) {
        throw std::runtime_error(std::string("state2::") + __func__ +
                                 " called after detached");
    }
    sync();
    progress_ += progress;
    check_resize();
}

void state2::operator++()
{
    (*this) += 1;
}

bool state2::should_stop()
{
    if (detached_) {
        return true;
    }
    sync();
    return detached_ ||
           (max_progress_.has_value() && progress_ >= max_progress_.value());
}

session &state2::sess()
{
    if (detached_) {
        throw std::runtime_error(std::string("state2::") + __func__ +
                                 " called after detached");
    }
    sync();
    return *sess_;
}

void state2::set_max_progress(int64_t n)
{
    if (detached_) {
        throw std::runtime_error(std::string("state2::") + __func__ +
                                 " called after detached");
    }
    sync();
    if (sess_->consistent(n)) {
        max_progress_ = n;
    }
}
}  // namespace stdml::collective
