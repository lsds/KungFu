#pragma once
#include <stddef.h>
#include <stdint.h>

#include <kungfu/callback.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void kungfu_run_main();
extern void kungfu_run_send_signal(int sig);

#ifdef __cplusplus
}

#include <kungfu/peer.hpp>
#include <kungfu/session.hpp>

namespace kungfu
{
extern int set_affinity(const Peer &peer);
}
#endif
