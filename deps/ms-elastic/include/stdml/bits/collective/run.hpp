#pragma once
#include <stdml/bits/collective/state2.hpp>
#include <stdml/bits/collective/state3.hpp>
#include <tracer/bits/stdtracer_xterm.hpp>

namespace stdml::collective
{
extern std::optional<cluster_config> go_get_cluster_config();

using init_fn = std::function<int64_t(session &, int64_t)>;
using sync_fn_t = std::function<void(session &, int64_t)>;
using step_fn_t = std::function<int64_t(session &, int64_t)>;

inline int64_t elastic_run(const init_fn &init, const sync_fn_t &sync,
                           const step_fn_t &run)
{
    int64_t max_progress = 0;
    state2 state(
        go_get_cluster_config,
        [&](session &sess, int64_t progress) {
            max_progress = init(sess, progress);
        },
        sync);

    state.set_max_progress(max_progress);

    for (; !state.should_stop();) {
        state += run(state.sess(), state);
    }
    return static_cast<int64_t>(state);
}

inline const char *ss(session &sess)
{
    static char line[32];
    sprintf(line, "<%d/%d>", (int)sess.rank(), (int)sess.size());
    return line;
}

using init_fn3 = std::function<int64_t(const session &, int64_t)>;

inline int64_t elastic_run3(const init_fn3 &init, const sync_fn_t &sync,
                            const step_fn_t &run)
{
    int64_t max_progress = 0;
    state3 state(
        go_get_cluster_config,
        [&](const session &sess, int64_t progress) {
            // fprintf(stderr, "calling user init %s\n", xt_green(ss(sess)));
            max_progress = init(sess, progress);
        },
        [&](session &sess, int64_t progress) {
            // fprintf(stderr, "user sync called %s\n", xt_green(ss(sess)));
            sync(sess, progress);
        });

    state.set_max_progress(max_progress);  // call private set

    for (; !state.should_stop();) {
        state += run(state.sess(), state);
    }
    return static_cast<int64_t>(state);
}
}  // namespace stdml::collective
