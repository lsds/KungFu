#pragma once
#ifdef KUNGFU_ENABLE_TRACE

#include <stdtracer.hpp>

using tracer_t = simple_tracer_t;

extern simple_tracer_ctx_t default_simple_ctx;
// extern log_tracer_ctx_t default_log_ctx;

#define DEFINE_TRACE_CONTEXTS simple_tracer_ctx_t default_simple_ctx("global");

#define TRACE_SCOPE(name)                                                      \
    tracer_t _((name), default_simple_ctx /*, default_log_ctx */)

#define TRACE_STMT(e)                                                          \
    {                                                                          \
        tracer_t _(#e, default_simple_ctx /*, default_log_ctx */);             \
        e;                                                                     \
    }

#define TRACE_EXPR(e)                                                          \
    [&]() {                                                                    \
        tracer_t _(#e, default_simple_ctx /*, default_log_ctx */);             \
        return (e);                                                            \
    }()

#else

#define TRACE_SCOPE(name)
// #define TRACE_STMT(e) e
// #define TRACE_EXPR(e) e

#define DEFINE_TRACE_CONTEXTS

#endif
