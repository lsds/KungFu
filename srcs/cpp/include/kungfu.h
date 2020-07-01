#pragma once
#include <stddef.h>
#include <stdint.h>

#include <kungfu/callback.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct order_group_s order_group_t;

extern order_group_t *new_ranked_order_group(int n_names);
extern void del_order_group(order_group_t *);
extern void order_group_do_rank(order_group_t *, int rank, callback_t *task);
extern void order_group_wait(order_group_t *, int32_t *arrive_order);

extern void kungfu_run_main();

#ifdef __cplusplus
}

#include <kungfu/peer.hpp>
#include <kungfu/session.hpp>
#endif
