#include "trace.hpp"

#ifdef TRACE_STACK

stack_tracer_ctx_t default_stack_tracer_ctx("global");

#else

simple_tracer_ctx_t default_simple_ctx("global");
// log_tracer_ctx_t default_log_ctx("global");

#endif
