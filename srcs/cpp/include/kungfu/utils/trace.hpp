#pragma once
#ifdef KUNGFU_ENABLE_TRACE

#include <stdtracer_thread>

#define DEFINE_TRACE_CONTEXTS DEFINE_TRACE_CONTEXT(global)

#else

#define TRACE_SCOPE(name)
// #define TRACE_STMT(e) e
// #define TRACE_EXPR(e) e

#define DEFINE_TRACE_CONTEXTS

#endif
