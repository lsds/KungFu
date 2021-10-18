#include <stdml/bits/collective/state3.hpp>
#include <tracer/bits/stdtracer_xterm.hpp>

/*
J   ->   S  <-> [C]
^        | \
|        v  \
P   <-   R   .-> F
^        |       |
|        v       v
I        D       Q
*/

static const xterm_t xt_red(1, 31);
static const xterm_t xt_green(1, 32);
static const xterm_t xt_yellow(1, 33);

namespace stdml::collective
{
static const char *s(session &sess)
{
    static char line[32];
    sprintf(line, "<%d/%d>", (int)sess.rank(), (int)sess.size());
    return line;
}

state3::state3(peer::config_prodiver get_config, init_state_fn init_state,
               sync_state_fn sync_state, std::optional<int64_t> max_progress)
    : peer_(peer::from_env()),
      sess_(peer_.join_elastic()),
      max_progress_(max_progress),
      progress_(0),
      synced_(false),
      detached_(false),
      get_config_(std::move(get_config)),
      sync_state_(std::move(sync_state))
{
    bool is_joined = sync_progress();
    if (init_state) {
        init_state(*sess_, progress_);
    }
    if (is_joined) {
        if (bool debug = false; debug) {
            fprintf(
                stderr,
                "calling user sync state at %d %s from state3::constructor\n",
                (int)progress_, xt_green(s(*sess_)));
        }
        sync_state_(*sess_, progress_);
    }
}

bool state3::sync_progress()
{
    if (!synced_) {
        progress_ = sess_->all_reduce(progress_, max);
        synced_ = true;
        return true;
    }
    return false;
}

void state3::check_resize()
{
    auto result = peer_.resize(sess_, get_config_);
    if (result.detached) {
        detached_ = true;
    }
    if (result.changed) {
        synced_ = false;
    }
}

void state3::sync()
{
    if (sync_progress()) {
        if (bool debug = false; debug) {
            fprintf(stderr,
                    "calling user sync state at %d %s from state3::sync\n",
                    (int)progress_, xt_green(s(*sess_)));
        }
        sync_state_(*sess_, progress_);
    }
}

state3::operator int64_t()
{
    if (detached_) {
        return progress_;
    }
    sync();
    return progress_;
}

void state3::operator+=(int64_t progress)
{
    if (detached_) {
        throw std::runtime_error(std::string("state3::") + __func__ +
                                 " called after detached");
    }
    sync();
    progress_ += progress;
    check_resize();
}

void state3::operator++()
{
    (*this) += 1;
}

bool state3::should_stop()
{
    if (detached_) {
        return true;
    }
    sync();
    return detached_ ||
           (max_progress_.has_value() && progress_ >= max_progress_.value());
}

session &state3::sess()
{
    if (detached_) {
        throw std::runtime_error(std::string("state3::") + __func__ +
                                 " called after detached");
    }
    sync();
    return *sess_;
}

void state3::set_max_progress(int64_t n)
{
    max_progress_ = n;
}
}  // namespace stdml::collective
